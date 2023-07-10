import argparse
import datetime
import logging
import os
import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from identify import util
class BaseTrainer:
    def __init__(self):
        self._device="cuda:0"
        name = str(datetime.datetime.now()).replace(' ', '_')
        self._save_path = os.path.join(self.args.save_path, name)

    def _save_model(self, save_path: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                    epoch: int,
                    name: str = 'model'):
        dir_path = os.path.join(save_path, name, str(epoch))
        util.create_directories_dir(dir_path)
        model.save_pretrained(dir_path)
        tokenizer.save_pretrained(dir_path)

