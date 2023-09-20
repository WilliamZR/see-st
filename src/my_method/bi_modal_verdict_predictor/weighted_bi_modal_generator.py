# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/10 19:38
# Description:

import os
import re

import pandas as pd
from torch.utils.data import Dataset

from my_methods.bi_modal_verdict_predictor.bi_modal_generator import BiModalGenerator
from my_utils import load_jsonl_data
from tqdm import tqdm
import torch

from utils.annotation_processor import AnnotationProcessor
from utils.prepare_model_input import init_db, prepare_input, remove_bracket_w_nonascii


def collate_fn(batch):
    raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, merge_weight, labels = map(list, zip(*batch))
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    token_type_ids = torch.stack(token_type_ids)
    input_ids2 = torch.stack(input_ids2)
    attention_mask2 = torch.stack(attention_mask2)
    merge_weight = torch.stack(merge_weight)
    labels = torch.stack(labels)
    return raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2, merge_weight, labels

label2weight = {
    "SENTENCE": [0.3, 0.7],
    "CELL":     [0.7, 0.3],
    "BOTH":     [0.5, 0.5],
}

class WeightedBiModalGenerator(BiModalGenerator):
    def __init__(self, input_path, tokenizer, cache_dir, data_type, args):
        super(WeightedBiModalGenerator, self).__init__(input_path, tokenizer, cache_dir, data_type, args)
        self.merge_weight = self.get_merge_weight(data_type)
        if len(self.merge_weight) != len(self.labels):
            self.merge_weight = self.merge_weight*2
        assert len(self.merge_weight) == len(self.labels)

    def get_merge_weight(self, data_type):
        data_type = data_type.split("_")[0]
        assert data_type in ["train", "dev","test"]
        input_path = f"{self.args.data_dir}/{data_type}.evi_type.jsonl"
        merge_weight = load_jsonl_data(input_path)
        merge_weight = [label2weight[mw] for mw in merge_weight]
        return merge_weight
    
    def __getitem__(self, idx):
        raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, labels \
            = super(WeightedBiModalGenerator, self).__getitem__(idx)
        merge_weight = torch.tensor(self.merge_weight[idx]).to(self.args.device)

        return raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, merge_weight, labels

    @classmethod
    def collate_fn(cls):
        return collate_fn

