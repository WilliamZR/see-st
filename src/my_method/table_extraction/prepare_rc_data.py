### rc_data_generator.py
### Wu Zirui
### 20221027

import argparse
import os
import re
import sys
import dgl
import pandas as pd
from tqdm import tqdm
import torch
from utils.util import JSONLineReader
import json
from utils.annotation_processor import AnnotationProcessor
from collections import defaultdict
from torch.utils.data.dataset import Dataset
import random
from my_utils import load_jsonl_data, load_pkl_data, save_jsonl_data, save_pkl_data
from baseline.drqa.retriever.doc_db import DocDB
import unicodedata
from utils.wiki_page import WikiPage


class RC_Generator(Dataset):
    def __init__(self, input_path, tokenizer, data_type, args):
        super(RC_Generator, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.input_path = input_path
        self.data_type = data_type
        self.raw_data = self.get_raw_data(input_path, data_type)

    def get_raw_data(self, input_path, data_type):
        jlr = JSONLineReader()
        file = '{0}/{1}.pos_neg_tables.jsonl'.format(input_path, data_type)
        with open(file, 'r') as f:
            lines = jlr.process(f)
        return lines

    def get_data_for_epoch(self, train = False):
        self.instances = []
        for entry in self.raw_data:
            if self.args.test_mode or not train:
                for cand in entry['all_candidates']:
                    cand['id'] = entry['id']
                    self.instances.append([entry['claim'], cand, None])
            else:
                pos_target = entry['pos_target']
                neg_target = entry['neg_target']
                if not pos_target or not neg_target:
                    continue
                for pt in pos_target:
                    nt = random.choice(neg_target)
                self.instances.append([entry['claim'], pt, nt])
        print('Dataset Constructed...')
        
    def get_pooling_matrix(self, token_type_ids, table_ids_2D):
        col_pooling_matrix = torch.zeros([len(table_ids_2D[0]), 512])
        row_pooling_matrix = torch.zeros([len(table_ids_2D), 512])
        for i in range(512):
            if token_type_ids[0, i, 0] == 0:
                continue
            elif token_type_ids[0, i, 0] == 1 and token_type_ids[0, i, 1] * token_type_ids[0, i, 2] > 0:
                col_num = token_type_ids[0,i,1] - 1
                row_num = token_type_ids[0,i,2] - 1
                col_pooling_matrix[col_num, i] = 1
                row_pooling_matrix[row_num, i] = 1

        row_pooling_matrix = average_of_matrix(row_pooling_matrix)
        col_pooling_matrix = average_of_matrix(col_pooling_matrix)

        return row_pooling_matrix, col_pooling_matrix

    def tokenize_claim_and_inputs(self, claim, cand):
        #    return {'table': table, 
        #    'table_ids': table_ids_2D, 
        #    'row_labels': row_labels,
        #    'col_labels': col_labels,
        #    'table_labels': table_label}
        if cand == None:
            tokenized_inputs = {'input_ids':[1,1],
                                'attention_mask': [2,2],
                                'token_type_ids': [1,1,1]}
            row_pooling_matrix = None
            col_pooling_matrix = None
            cand = {'row_labels': None,
                    'col_labels': None,
                    'table_labels': None}
        else:
            table =pd.DataFrame(cand['table'], columns=cand['table'][0], dtype = str).fillna('')
            tokenized_inputs = self.tokenizer(table = table, queries = claim, padding = 'max_length', truncation = True, return_tensors = 'pt')
            row_pooling_matrix, col_pooling_matrix = self.get_pooling_matrix(tokenized_inputs['token_type_ids'], cand['table_ids'])
        
        return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], tokenized_inputs['token_type_ids'], row_pooling_matrix, col_pooling_matrix\
            , cand['row_labels'], cand['col_labels'], cand['table_labels']


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx, mode = 'tapas'):
        ## [claim, pos_cand, neg_cand]
        data = self.instances[idx]
        if mode == 'tapas':
            pos_ids, pos_masks, pos_token_ids, pos_row_pooling_matrix, pos_col_pooling_matrix\
                , pos_row_labels, pos_col_labels, pos_table_label = self.tokenize_claim_and_inputs(data[0], data[1])
            neg_ids, neg_masks, neg_token_ids, neg_row_pooling_matrix, neg_col_pooling_matrix\
                , neg_row_labels, neg_col_labels, neg_table_label = self.tokenize_claim_and_inputs(data[0], data[2])

            return pos_ids, pos_masks, pos_token_ids\
                , neg_ids, neg_masks, neg_token_ids\
                , pos_row_pooling_matrix, pos_col_pooling_matrix\
                , neg_row_pooling_matrix, neg_col_pooling_matrix\
                , pos_row_labels, pos_col_labels\
                , neg_row_labels, neg_col_labels
        else:
            raise NotImplementedError

def get_all_tables(page):
    page = unicodedata.normalize('NFD', page)
    try:
        lines = json.loads(db.get_doc_json(page))
    except:
        return []

    current_page = WikiPage(page, lines)
    all_tables = current_page.get_tables()

    return all_tables


def get_candidates(tab, gold_cell_evidence):
    row_labels = set()
    col_labels = set()
    table_content_2D = []
    table_ids_2D = []
    page = tab.page
    for i, row in enumerate(tab.rows):
        if i == 128:
            break
        row_flat = []
        row_id = []
        for j, cell in enumerate(row.row):
            if j == 128:
                break
            curr_id = page + '_' + cell.get_id()
            if curr_id in gold_cell_evidence:
                col_labels.add(j)
                row_labels.add(i)
            row_flat.append(str(cell))
            row_id.append(curr_id)
        table_ids_2D.append(row_id)
        table_content_2D.append(row_flat)
    table_label = 1 if len(row_labels) + len(col_labels) > 1 else 0
    col_labels = [1 if i in col_labels else 0 for i in range(len(table_ids_2D[0]))]
    row_labels = [1 if i in row_labels else 0 for i in range(len(table_ids_2D))]
    #table = pd.DataFrame(table_content_2D, columns = table_content_2D, dtype = str).fillna(' ')
    return {'table': table_content_2D, 
            'table_ids': table_ids_2D, 
            'row_labels': row_labels,
            'col_labels': col_labels,
            'table_labels': table_label}


def generate_table_pair_line(line, gold_line, args):
    id = gold_line["id"]
    claim = gold_line["claim"]
    all_pos_candidates = []
    all_neg_candidates = []
    all_candidates = []
    gold_table_evidence =set()
    gold_cell_evidence = set()
    gold_evi_by_page = defaultdict(list)

    if 'evidence' in gold_line:
        gold_cell_evidence = [evi for evi_set in gold_line["evidence"] for evi in evi_set["content"] if "_cell_" in evi]
        gold_table_evidence = set(evi.split('_')[0] + '_table_' + evi.split('_')[-3] for evi in gold_cell_evidence)

        for evi in gold_table_evidence:
            title = evi.split('_')[0]
            table_id = '_'.join(evi.split('_')[1:])
            gold_evi_by_page[title].append(table_id)

        for page in gold_evi_by_page:
            tables = get_all_tables(page)
            if not tables:
                continue
            for tab in tables:
                instance = get_candidates(tab, gold_cell_evidence)
                all_pos_candidates.append(instance)
    else:
        gold_cell_evidence = []

    if 'predicted_pages' in line:
        sorted_p = list(sorted(line['predicted_pages'], reverse=True, key=lambda elem: elem[1]))
        pages = [p[0] for p in sorted_p[:args.max_page]]
        for page in pages:
            tables = get_all_tables(page)
            for tab in tables:
                instance = get_candidates(tab, gold_cell_evidence)
                all_candidates.append(instance)
                if page in gold_table_evidence:
                    continue
                all_neg_candidates.append(instance)
    output = {
        'id': id,
        'claim': claim,
        'pos_target': all_pos_candidates,
        'neg_target': all_neg_candidates,
        'all_candidates': all_candidates
    }
    return output
            
def average_of_matrix(pooling_matrix):
    average_num = torch.sum(pooling_matrix, dim = 1)
    average_num = torch.reshape(average_num, (-1, 1))
    average_num = torch.where(average_num == 0, torch.ones_like(average_num), average_num)
    pooling_matrix = torch.div(pooling_matrix, average_num)
    return pooling_matrix
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type = str, default='data/feverous_wikiv1.db')
    parser.add_argument('--data_path', type = str, default = 'data')
    parser.add_argument('--max_page', type = int, default = 5)
    parser.add_argument('--force_generate', action = 'store_true')
    args = parser.parse_args()
    print(args)
    db = DocDB(args.db)

    for split in ['train', 'dev', 'test']:
        jlr = JSONLineReader()

        gold_input_path = '{0}/{1}.jsonl'.format(args.data_path, split)
        gold_data = load_jsonl_data(gold_input_path)[1:]
        input_path = "{0}/{1}.pages.p{2}.jsonl".format(args.data_path, split, args.max_page)
        output_path = '{0}/{1}.pos_neg_tables.jsonl'.format(args.data_path, split)

        if os.path.exists(output_path) and not args.force_generate:
            print("File for Table Extraction exists: {}".format(output_path))
            continue
        with open(input_path, 'r') as f, open(output_path, 'w') as writer:
            lines = jlr.process(f)
            for line, gold_line in tqdm(zip(lines, gold_data)):
                line = generate_table_pair_line(line, gold_line, args)
                writer.write(json.dumps(line) + "\n")


