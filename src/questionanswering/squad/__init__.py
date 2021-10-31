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
    res = qa.preprocess(s)
    res = re.sub(r"\b(religion)(note)\b", r"\1 \2", res)
    res = re.sub(r"\b(equanimity)(full)\b", r"\1 \2", res)
    return res


def nearest(s: str, t: str, start: int) -> int:
    """Returns the beginning index of the answer span nearest to start index.
    If answers spans are found at equal distance on the left and right, return the left.
    """
    gap = len(t)
    best = -1
    ps = re.escape(s)
    if s[0].isalnum():
        ps = r"\b" + ps
    if s[-1].isalnum():
        ps += r"\b"
    p = re.compile(ps, re.IGNORECASE)
    for m in p.finditer(t):
        d = abs(m.start() - start)
        if d < gap:
            gap = d
            best = m.start()
    return best


ALTERNATIVE_ANSWERS: Dict[str, List[str]] = {
    "1": ["one", "first", "1st", "single"],
    "2": ["two", "second", "2nd", "double", "ii"],
    "3": ["three", "third", "3rd", "triple", "iii"],
    "4": ["four", "fourth", "4th"],
    "5": ["five", "fifth", "5th"],
    "6": ["six", "sixth", "6th"],
    "7": ["seven", "seventh", "7th"],
    "8": ["eight", "eighth", "8th"],
    "9": ["nine", "ninth", "9th"],
    "10": ["ten", "tenth", "10th"],
    "11": ["eleven", "eleventh", "11th"],
    "12": ["twelve", "twelfth", "12th"],
    "13": ["thirteen", "thirteenth", "13th"],
    "14": ["fourteen", "fourteenth", "14th"],
    "15": ["fifteen", "fifteenth", "15th"],
    "16": ["sixteen", "sixteenth", "16th"],
    "17": ["seventeen", "seventeenth", "17th"],
    "18": ["eighteen", "eighteenth", "18th"],
    "19": ["nineteen", "nineteenth", "19th"],
    "20": ["twenty", "twentieth", "20th"],
    "24": ["twenty - four", "twenty - fourth", "24th"],
    "41": ["forty - one", "forty - first", "41st"],
    "four": ["fourth"],
    "six": ["sixth"],
    "seven": ["seventh"],
    "ten": ["tenth"],
    "thirteen": ["thirteenth"],
    "fourteen": ["fourteenth"],
    "twenty - four": ["twenty - fourth"],
    "north": ["northern"],
    "south": ["southern"],
    "east": ["eastern"],
    "west": ["western"],
    "northeast": ["northeastern"],
    "northwest": ["northwestern"],
    "southeast": ["southeastern"],
    "southwest": ["southwestern"],
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
    "56d1c2d2e7d4791d00902121": "5th century ce",
    "56db1d2fe7c41114004b4d68": "hairdo",
    "56d3883859d6e41400146678": "almost a decade",
    "56d38b4e59d6e414001466d9": "2011",
    "56d5f9181c85041400946e7d": "fearlessness",
    "56d5fc2a1c85041400946ea0": "breeds",
    "56d62e521c85041400946f9f": "emotional",
    "56d62f3e1c85041400946fa5": "hunting",
    "56de0abc4396321400ee2563": "islamic",
    "56df844f56340a1900b29cca": "700 lumens",
    "56df865956340a1900b29ceb": "daylight factor calculation",
    "56df95d44a1a83140091eb81": "the greater the apparent saturation or vividness of the object colors",
    "56dfa6de7aa994140058df9a": "shorter postoperative hospital stays , received fewer negative evaluative comments in nurses' notes , and took fewer potent analegesics",
    "56dfa6de7aa994140058df9b": "1972 and 1981",
    "56e4793839bdeb140034794f": "constructed in a manner which is environmentally friendly",
    "56d4baf92ccc5a1400d8317f": "destiny fulfilled",
    "56d0e42e17492d1400aab68c": "he lived , taught and founded a monastic order",
    "56de4adf4396321400ee278e": "blue dashes",
    "56de8542cffd8e1900b4b9da": "western european powers",
    "56df2305c65bf219000b3f98": "general electric",
    "56df5e8e96943c1400a5d44d": "tinker air force base",
    "56dfb5977aa994140058e02d": "1907–1912",
    "56df736f5ca0a614008f9a91": "30",
}

QUESTION_REPLACEMENTS: Dict[str, str] = {
    "56cc57466d243a140015ef24": "in which year did the sales of iPhone exceed iPod?",
    "56d3883859d6e41400146678": "For how long was American Idol the highest rated reality show on television?",
    "56d38b4e59d6e414001466d9": "In which year did American Idol win an award for Best Reality Competition?",
    "56df865956340a1900b29ceb": "What technique considers the amount of daylight received indoors?",
    "56df95d44a1a83140091eb81": "What is the effect of high GAI value?",
    "56dfa6de7aa994140058df9a": "What are the medical outcomes for patients with natural scenery?",
    "56dfa6de7aa994140058df9b": "when was the study conducted by robert ulrich?",
    "56e4793839bdeb140034794f": "How should a building fulfil the contemporary ethos?",
    "56d0e42e17492d1400aab68c": "What details of buddha's life that most scholars accept?",
    "56df5e8e96943c1400a5d44d": "who is the biggest employer in the msa area?",
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
