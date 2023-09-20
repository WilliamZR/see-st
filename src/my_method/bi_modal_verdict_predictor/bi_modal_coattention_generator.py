# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/3 16:06
# Description:

# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2021/12/14 22:36
# Description:
import os
import re

import pandas as pd
from torch.utils.data import Dataset
import bi_modal_config as config
from base_templates import BaseGenerator
from tqdm import tqdm
import torch
from my_utils.common_utils import load_pkl_data, save_pkl_data

from utils.annotation_processor import AnnotationProcessor
from utils.prepare_model_input import init_db, prepare_input, remove_bracket_w_nonascii
from bi_modal_arg_parser import get_parser  
from base_templates import BasePreprocessor

from transformers import AutoTokenizer, RobertaTokenizer


def collate_fn(batch):
    raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, labels = map(list, zip(*batch))
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    token_type_ids = torch.stack(token_type_ids)
    input_ids2 = torch.stack(input_ids2)
    attention_mask2 = torch.stack(attention_mask2)
    labels = torch.stack(labels)
    return raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2, labels


class BiModalCoAttentionGenerator(BaseGenerator):
    def __init__(self, input_path, tokenizer, cache_dir, data_type, args):
        super(BiModalCoAttentionGenerator, self).__init__(input_path, data_type, args)
        init_db(args.wiki_path)
        self.tokenizer = tokenizer
        self.data_type = data_type
        if data_type in ["train"]:
            claim_evidence_input, evidence_input, annotations = self.get_train_data()
        elif data_type in ["dev_retrieved", "valid", "development", "dev", "validation", "test"]:
            claim_evidence_input, evidence_input, annotations = self.get_test_data(retrieved = (("retrieved" in data_type) or ("test" in data_type)))
        else:
            assert False, data_type

        self.labels = [self.config.label2idx[anno.get_verdict()] if hasattr(anno, "verdict") else 0 for anno in annotations]
        if self.data_type != "test":
            self.print_label_distribution(self.labels)
        assert self.labels != 0
        self.claims = [anno.claim for anno in annotations]
        self.flatten_tables = process_data(evidence_input)
        self.claim_evidence_input = process_data(claim_evidence_input)

        assert len(self.labels) != 0
        assert len(self.claims) == len(self.labels)
        assert len(self.flatten_tables) == len(self.labels)
        assert len(self.claim_evidence_input) == len(self.labels)
        assert len(self.raw_data) == len(self.labels)

    def print_label_distribution(self, labels_train):
        NEI_num = len([lb for lb in labels_train if lb == 0])
        SUP_num = len([lb for lb in labels_train if lb == 1])
        REF_num = len([lb for lb in labels_train if lb == 2])
        print(f"NEI: {NEI_num * 1.0 / len(labels_train)}, SUPPORTS: {SUP_num * 1.0 / len(labels_train)}, "
              f"REFUTES: {REF_num * 1.0 / len(labels_train)}")

    def get_refine_keys(self):
        keys = ["id", "claim", "evidence", "predicted_evidence", "label"]
        return keys

    def get_train_data(self):
        args = self.args
        all_evidence_input = []
        all_claim_evidence_input = []
        all_annotations_train = []
        self.labels = []
        self.raw_data = []

        assert args.train_data_type in ["gold", "retrieved", "both", "fusion"]
        if not self.args.cache_name:    
            cache_path = os.path.join('data/', 'train.{}.bimodal.coattention.cache.json'.format(args.train_data_type))
        else:
            cache_path = os.path.join('data/', 'train.{}.json'.format(self.args.cache_name))
        if os.path.exists(cache_path) and not args.force_generate:
            cache_data = load_pkl_data(cache_path)
            all_claim_evidence_input = cache_data[0]
            all_evidence_input = cache_data[1]
            all_annotations_train = cache_data[2]
            if args.train_data_type == "gold" or args.train_data_type == "both":
                args.train_data_path = os.path.join(args.data_dir, 'train.jsonl')
                raw_data = self.preprocess_raw_data(self.get_raw_data(args.train_data_path, keys=self.get_refine_keys()))
                self.raw_data.extend(raw_data)
            if args.train_data_type == "retrieved" or args.train_data_type == "both" or args.train_data_type == "fusion": 
                args.train_data_path = os.path.join(args.data_dir, 'train.combined.not_precomputed.p5.s5.t3.cells.jsonl')
                if args.train_data_type == "fusion":
                    if not args.input_name:
                        args.train_data_path = os.path.join(args.data_dir, 'train.scorecheck.jsonl')
                    else:
                        args.train_data_path = os.path.join(args.data_dir, 'train.' + args.input_name + '.jsonl')
                raw_data = self.preprocess_raw_data(self.get_raw_data(args.train_data_path, keys=self.get_refine_keys()))
                self.raw_data.extend(raw_data) 
        else:
            if args.train_data_type == "gold" or args.train_data_type == "both":
                args.train_data_path = os.path.join(args.data_dir, 'train.jsonl')
                anno_processor_train = AnnotationProcessor(args.train_data_path, has_content=True)
                annotations_train = [annotation for annotation in anno_processor_train]
                evidence_input = [(prepare_input(anno, 'all2tab', gold=True), anno.get_verdict()) for i, anno in
                                        enumerate(tqdm(annotations_train))]
                all_evidence_input.extend(evidence_input)
                claim_evidence_input = [(prepare_input(anno, 'all2text', gold=True), anno.get_verdict()) for i, anno in
                                        enumerate(tqdm(annotations_train))]
                all_claim_evidence_input.extend(claim_evidence_input)
                all_annotations_train.extend(annotations_train)
                raw_data = self.preprocess_raw_data(self.get_raw_data(args.train_data_path, keys=self.get_refine_keys()))
                self.raw_data.extend(raw_data)

            if args.train_data_type == "retrieved" or args.train_data_type == "both" or args.train_data_type == "fusion":
                args.train_data_path = os.path.join(args.data_dir, 'train.combined.not_precomputed.p5.s5.t3.cells.jsonl')
                if args.train_data_type == "fusion":
                    if args.train_data_type == "fusion":
                        if not args.input_name:
                            args.train_data_path = os.path.join(args.data_dir, 'train.scorecheck.jsonl')
                        else:
                            args.train_data_path = os.path.join(args.data_dir,  'train.' + args.input_name + '.jsonl')
                print("training data path:", args.train_data_path)
                anno_processor_train = AnnotationProcessor(args.train_data_path, has_content=True)
                annotations_train = [annotation for annotation in anno_processor_train]
                if args.revise_labels:
                    for anno in annotations_train:
                        if anno.verdict == "NOT ENOUGH INFO":
                            continue
                        predicted_evi = set(anno.predicted_evidence)
                        gold_evi_lst = anno.evidence
                        sufficient = False
                        for gold_evi in gold_evi_lst:
                            if set(gold_evi).issubset(predicted_evi):
                                sufficient = True
                                break
                        if not sufficient:
                            # print(anno.verdict, "==> NEI")
                            anno.verdict = "NOT ENOUGH INFO"
                evidence_input = [(prepare_input(anno, 'all2tab', gold=False), anno.get_verdict()) for i, anno in
                                        enumerate(tqdm(annotations_train))]
                all_evidence_input.extend(evidence_input)
                claim_evidence_input = [(prepare_input(anno, 'all2text', gold=False), anno.get_verdict()) for i, anno in
                                        enumerate(tqdm(annotations_train))]
                all_claim_evidence_input.extend(claim_evidence_input)
                all_annotations_train.extend(annotations_train)
                raw_data = self.preprocess_raw_data(self.get_raw_data(args.train_data_path, keys=self.get_refine_keys()))
                self.raw_data.extend(raw_data)
                save_pkl_data([all_claim_evidence_input,
                                all_evidence_input,
                                all_annotations_train], cache_path)
        print("training data path:", args.train_data_path)

        return all_claim_evidence_input, all_evidence_input, all_annotations_train

    def get_test_data(self, retrieved = False):
        args = self.args
        if self.data_type == "test":
            file_name = 'test.scorecheck.jsonl'
        else:
            #file_name = "dev.scorecheck.jsonl" if retrieved else 'dev.jsonl'
            if not retrieved:
                file_name = "dev.jsonl"
            else:
                data_type = 'dev' if 'dev' in self.data_type else 'test'
                if not self.args.input_name:
                    file_name = 'dev.scorecheck.jsonl'
                else:
                    file_name = data_type + '.' + args.input_name + '.jsonl'
        if not self.args.cache_name:
            cache_path = os.path.join('data/', '{}.bimodal.coattention.cache.json'.format(self.data_type))
        else:
            cache_path = os.path.join('data/', '{}.{}.json'.format(self.data_type, self.args.cache_name))
        if os.path.exists(cache_path) and not args.force_generate:
            cache_data = load_pkl_data(cache_path)
            claim_evidence_input_test = cache_data[0]
            evidence_input_test = cache_data[1]
            annotations_dev = cache_data[2]
        else:
            args.dev_data_path = os.path.join(args.data_dir, file_name)
            anno_processor_dev = AnnotationProcessor(args.dev_data_path, has_content=False)
            annotations_dev = [annotation for annotation in anno_processor_dev]
            evidence_input_test = [(prepare_input(anno, 'all2tab', gold=(not retrieved)), anno.get_verdict()) for i, anno in
                                        enumerate(tqdm(annotations_dev))]
            claim_evidence_input_test = [(prepare_input(anno, 'all2text', gold=(not retrieved)), anno.get_verdict()) for
                                        i, anno in
                                        enumerate(tqdm(annotations_dev))]
            save_pkl_data([claim_evidence_input_test,
                           evidence_input_test,
                           annotations_dev],
                           cache_path)
        print("test data path:", cache_path)
        return claim_evidence_input_test, evidence_input_test, annotations_dev

    def preprocess_raw_data(self, raw_data):
        if raw_data[0]["claim"]:
            return raw_data
        else:
            return raw_data[1:]

    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        encodings = self.tokenizer[0](table=read_text_as_pandas_table(self.flatten_tables[idx]), queries=self.claims[idx]
                                   , padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings["input_ids"]).to(self.args.device)
        attention_mask = torch.tensor(encodings["attention_mask"]).to(self.args.device)
        token_type_ids = torch.tensor(encodings["token_type_ids"]).to(self.args.device)

        encodings = self.tokenizer[1](self.claim_evidence_input[idx], padding="max_length", truncation=True)
        input_ids2 = torch.tensor(encodings["input_ids"]).to(self.args.device)
        attention_mask2 = torch.tensor(encodings["attention_mask"]).to(self.args.device)
        labels = torch.tensor(self.labels[idx]).to(self.args.device)

        return raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, labels

    @classmethod
    def collate_fn(cls):
        return collate_fn

def process_data(claim_verdict_list):
    text = [x[0] for x in claim_verdict_list]#["I love Pixar.", "I don't care for Pixar."]
    pt = re.compile(r"\[\[.*?\|(.*?)]]")
    # Fix bug for code below expected string or bytes-like object
  
    text = [re.sub(pt, r"\1", text) for text in text]
    
    
    # text_test = [re.sub(pt, r"[ \1 ]", text) for text in text_test]
    text = [remove_bracket_w_nonascii(text) for text in text]

    return text

def read_text_as_pandas_table(table_text: str):
    table = pd.DataFrame([x.split(' | ') for x in table_text.split('\n')][:255], columns=[x for x in table_text.split('\n')[0].split(' | ')]).fillna('')
    table = table.astype(str)
    return table




def get_sent_index_and_mask(input_ids):
    sent_index = []
    for idx in input_ids:
        sent_index.append([i for i in range(len(idx)) if idx[i] == 2])
    max_sent = max([len(idx) for idx in sent_index])
    sent_mask = torch.tensor([[1] * len(idx) + [0] * (max_sent - len(idx)) for idx in sent_index])
    sent_index = torch.tensor([idx + [0] * (max_sent - len(idx)) for idx in sent_index])
    return sent_index, sent_mask

def get_table_index_and_mask(row_index):
    max_row = max([len(idx) for idx in row_index])
    table_mask = torch.tensor([[1] * len(idx) + [0] * (max_row - len(idx)) for idx in row_index])
    table_index = torch.tensor([idx + [0] * (max_row - len(idx)) for idx in row_index])
    return table_index, table_mask

def get_row_index(table, token_type_ids):
    row_set = set()
    for i in range(len(table)):
    ##第一个单元格字符
        cell = table.iloc[i,0]
        if cell.startswith('[T]'):
            row_set.add(i+1)
    row_index_list = []
    for i in range(token_type_ids.size()[-2]):
        if int(token_type_ids[i,0]) == 1 and int(token_type_ids[i,1]) == 1 and int(token_type_ids[i,2]) in row_set:
            row_index_list.append(i)
            row_set.remove(int(token_type_ids[i,2]))
    ## assert, if code stops, print row_index_list, row_index_list
    return row_index_list

def hierarchical_collate_fn(batch):
    raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, row_index, labels = map(list, zip(*batch))
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    token_type_ids = torch.stack(token_type_ids)
    input_ids2 = torch.stack(input_ids2)
    attention_mask2 = torch.stack(attention_mask2)
    labels = torch.stack(labels)
    sent_index, sent_mask = get_sent_index_and_mask(input_ids2)
    table_index, table_mask = get_table_index_and_mask(row_index)
    return raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2,\
        table_index, table_mask, sent_index, sent_mask, labels

class BiModalHierarchialCoAttentionGenerator(BiModalCoAttentionGenerator):
    def __init__(self, input_path, tokenizer, cache_dir, data_type, args):
        super().__init__(input_path, tokenizer, cache_dir, data_type, args)

    @classmethod
    def collate_fn(cls):
        return hierarchical_collate_fn
    
    
    def __getitem__(self, idx):
        raw_data = self.raw_data[idx]
        curr_tab = read_text_as_pandas_table(self.flatten_tables[idx])
        encodings = self.tokenizer[0](table= curr_tab, queries=self.claims[idx]
                                   , padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings["input_ids"]).to(self.args.device)
        attention_mask = torch.tensor(encodings["attention_mask"]).to(self.args.device)
        token_type_ids = torch.tensor(encodings["token_type_ids"]).to(self.args.device)

        encodings = self.tokenizer[1](self.claim_evidence_input[idx], padding="max_length", truncation=True)
        input_ids2 = torch.tensor(encodings["input_ids"]).to(self.args.device)
        attention_mask2 = torch.tensor(encodings["attention_mask"]).to(self.args.device)
        labels = torch.tensor(self.labels[idx]).to(self.args.device)
        row_index = get_row_index(curr_tab, token_type_ids)
        return raw_data, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, row_index, labels


if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)
    
    # 1 table 2 text
    args.bert_name_2 = '/home/hunan/bert_weights/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer2 = RobertaTokenizer.from_pretrained(args.bert_name_2, model_max_length=512)
    preprocessor = BasePreprocessor(args)
    args.tokenizer = [tokenizer, tokenizer2]
    args.config = config
    data_generator = BiModalCoAttentionGenerator
    train_data, valid_data, test_data = preprocessor.process(args.data_dir, args.cache_dir
                                                    , data_generator, args.tokenizer, dataset=["dev_retrieved", "train"])