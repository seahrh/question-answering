import re
import json
import pathlib
from typing import Tuple, List, Union, Dict, Set

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForQuestionAnswering, AdamW

import questionanswering as qa


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
    res = re.sub(r"full citation needed\b", r"", res)
    res = re.sub(r"(citation|clarification) needed\b", r"", res)
    res = re.sub(r"\b(\w+)(note)\b", r"\1 \2", res)
    res = re.sub(r"\b(\w+)(update)\b", r"\1 \2", res)
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
    "1": ["one", "first", "single"],
    "2": ["two", "second", "double", "ii"],
    "3": ["three", "third", "triple", "iii"],
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
    "21": ["twenty - one", "twenty - first"],
    "24": ["twenty - four", "twenty - fourth"],
    "41": ["forty - one", "forty - first"],
    "four": ["fourth"],
    "six": ["sixth"],
    "seven": ["seventh"],
    "ten": ["tenth"],
    "thirteen": ["thirteenth"],
    "fourteen": ["fourteenth"],
    "twenty - four": ["twenty - fourth"],
}

ANSWER_CORRECTIONS: Dict[str, str] = {
    "56cc57466d243a140015ef24": "2010",
    "56cebbdeaab44d1400b8895c": "a million",
    "56d1c2d2e7d4791d00902121": "5th century ce",
    "56db1d2fe7c41114004b4d68": "hairdo",
    "56d3883859d6e41400146678": "almost a decade",
    "56d38b4e59d6e414001466d9": "2011",
    "56df844f56340a1900b29cca": "700 lumens",
    "56df865956340a1900b29ceb": "daylight factor calculation",
    "56df95d44a1a83140091eb81": "the greater the apparent saturation or vividness of the object colors",
    "56dfa6de7aa994140058df9a": "shorter postoperative hospital stays , received fewer negative evaluative comments in nurses' notes , and took fewer potent analegesics",
    "56dfa6de7aa994140058df9b": "1972 and 1981",
    "56e4793839bdeb140034794f": "constructed in a manner which is environmentally friendly",
    "56d4baf92ccc5a1400d8317f": "destiny fulfilled",
    "56d0e42e17492d1400aab68c": "he lived , taught and founded a monastic order",
    "56de8542cffd8e1900b4b9da": "western european powers",
    "56df2305c65bf219000b3f98": "general electric",
    "56df5e8e96943c1400a5d44d": "tinker air force base",
    "56dfb5977aa994140058e02d": "1907–1912",
    "56df736f5ca0a614008f9a91": "30",
    "56e042487aa994140058e409": "⟨p⟩",
    "56e0bc7b231d4119001ac364": "december 6 , 1957",
    "56e83bdf37bdd419002c44be": "ser",
    "56f7194f3d8e2e1400e3734c": "the word slovo word and the related slava fame and slukh hearing",
    "56f7d4f7aef2371900625c25": "hereditary juridical status",
    "56f852d0a6d7ea1400e17569": "not all landed gentry had a hereditary title of nobility",
    "56fdcd6019033b140034cd8b": "1950s",
    "5706300775f01819005e7a62": "sync word",
    "570cee7ffed7b91900d45afe": "infected plant cells form crown gall or root tumors",
    "570ce1bab3d812140066d2e2": "the start value of a routine is based on the difficulty of the elements the gymnast attempts and whether or not the gymnast meets composition requirements",
    "570ce94dfed7b91900d45ad0": "external force which the gymnasts have to overcome with their muscle force and has an impact on the gymnasts linear and angular momentum",
    "570dfa320b85d914000d7c48": "varies widely between jurisdictions",
    "570ffff5b654c5140001f725": "men did not show any sexual arousal to non - human visual stimuli",
    "571015bea58dae1900cd6877": "overcome immense difficulties in characterizing the sexual orientation of trans men and trans women",
    "571a275210f8ca1400304f06": "information from the outside world to be sensed in the form of chemical and physical stimuli",
    "57262473271a42140099d4ed": "the hellenistic period",
    "5725bc3989a1e219009abda6": "us $ 5 million grant for the international justice mission ijm",
    "57265693dd62a815002e81ed": "dispense with the appearance of neutrality and use their influence to unfairly influence the outcome of the match for added dramatic impact",
    "57292b563f37b319004780b1": "battle of the teutoburg forest",
    "57265e2b5951b619008f70b6": "florida had become a derelict open to the occupancy of every enemy",
    "572670f2f1498d1400e8dfb6": "first post - reconstruction republican governor , in an upset election",
    "572670f2f1498d1400e8dfb7": "a white conservative , was elected as the state's first post - reconstruction republican us senator",
    "5728f8422ca10214002dab4c": "increasing numbers of airlines have began launching direct flights from japan , qatar , taiwan , south korea , germany and singapore",
    "572908166aef0514001549cb": "privately funded english language schools",
    "5726d6b55951b619008f7f8e": "place half of poland border along the vistula river , latvia , estonia , finland , and bessarabia in the soviets' sphere of influence",
    "5726d9dcdd62a815002e9296": "two",
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
    "56f7d4f7aef2371900625c25": "what confers nobility?",
    "57262473271a42140099d4ed": "when was the peak of greek cultural influence and power?",
    "5728f8422ca10214002dab4c": "has the travel industry considered making changes to flight plans for myanmar?",
    "5726d6b55951b619008f7f8e": "what did the soviets get out of the non - aggression pact?",
}

ANSWER_SUFFIXES: Set[str] = {
    "an",  # european
    "ary",  # evolutionary
    "i",  # israeli
    "izing",  # localizing
    "ist",  # motorist
    "ity",  # uniformity
    "s",
    "ly",
    "ed",  # played
    "er",
    "ern",  # northern
    "es",
    "ese",  # japanese
    "al",
    "nd",
    "ness",
    "tic",
    "rd",
    "son",
    "st",
    "th",
    "ing",
    "ic",
    "time",
    "ward",
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
                        for alt in ALTERNATIVE_ANSWERS[at]:
                            i = nearest(
                                s=alt,
                                t=context,
                                start=a["answer_start"],
                            )
                            if i >= 0:
                                at = alt
                                break
                    if i == -1:
                        i = nearest(s=at, t=context, start=a["answer_start"])
                    # try suffix "ly" e.g. "extremely"
                    if i == -1:
                        for suffix in ANSWER_SUFFIXES:
                            alt = at + suffix
                            i = nearest(s=alt, t=context, start=a["answer_start"])
                            if i >= 0:
                                at = alt
                                break
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
