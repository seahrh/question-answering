import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForQuestionAnswering, AdamW, BatchEncoding
from typing import Tuple, List
import questionanswering as qa

__all__ = ["Dataset", "Model", "position_labels"]

log = qa.get_logger()


def position_labels(
    encodings: BatchEncoding,
    answer_start: List[int],
    answer_end: List[int],
    ids: List[int],
    is_impossible: List[int],
) -> Tuple[List[int], List[int]]:
    start_positions = []
    end_positions = []
    for i in range(len(is_impossible)):
        j, k = 0, 0
        if is_impossible[i] == 0:
            j = encodings.char_to_token(i, char_index=answer_start[i], sequence_index=1)
            if j is None:
                raise ValueError(
                    f"start pos must not be None. i={i}, id={ids[i]}, answer_start={answer_start[i]}"
                )
            k = encodings.char_to_token(
                i, char_index=answer_end[i] - 1, sequence_index=1
            )
            if k is None:
                raise ValueError(
                    f"end pos must not be None. i={i}, id={ids[i]}, answer_end={answer_end[i]}"
                )
            if j > k:
                raise ValueError("start pos must be less than or equals end pos")
        start_positions.append(j)
        end_positions.append(k)
    return start_positions, end_positions


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class Model(pl.LightningModule):
    def __init__(self, pretrained_dir: str, gradient_checkpointing: bool, lr: float):
        super().__init__()
        self.lr = lr
        config = AutoConfig.from_pretrained(pretrained_dir)
        config.gradient_checkpointing = gradient_checkpointing
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_dir, config=config
        )
        self.register_buffer("start_logits", torch.zeros(1))
        self.register_buffer("end_logits", torch.zeros(1))

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
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
        optimizer = AdamW(self.parameters(), lr=self.lr, correct_bias=True)
        return optimizer
