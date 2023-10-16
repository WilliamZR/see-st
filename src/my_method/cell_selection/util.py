### util.py
### Wu Zirui
### 20221201
### 

import pandas as pd
from collections import defaultdict
import sys
import jsonlines
from my_utils.common_utils import load_pkl_data, save_pkl_data
import pickle
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel, TapasTokenizer
import argparse
from tqdm import tqdm
import os
import re
import torch
import random
from utils.annotation_processor import AnnotationProcessor, EvidenceType
from utils.wiki_page import WikiPage, get_wikipage_by_id,WikiTable
from database.feverous_db import FeverousDB
import os

from graph_parser import get_parser
from baseline.drqa.tokenizers.spacy_tokenizer import SpacyTokenizer
TOKENIZER = SpacyTokenizer(annotators=set(['ner']))
from torchnet import meter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

def average_of_matrix(pooling_matrix):
    average_num = torch.sum(pooling_matrix, dim = 1)
    average_num = torch.reshape(average_num, (-1, 1))
    average_num = torch.where(average_num == 0, torch.ones_like(average_num), average_num)
    pooling_matrix = torch.div(pooling_matrix, average_num)
    return pooling_matrix

def average(my_list):
    return sum(my_list) / len(my_list)

def reshape_labels(labels):
    if len(labels) == 1:
        return labels
    output =  [i for item in labels for i in item]
    return output

def reshape_input(table_input_ids, table_attention_masks, table_token_type_ids):
    if len(table_input_ids) == 1:
        return table_input_ids[0], table_attention_masks[0], table_token_type_ids[0]
    table_input_ids = [torch.reshape(item, (-1, 512)) for item in table_input_ids if item != []]
    table_attention_masks = [torch.reshape(item, (-1, 512)) for item in table_attention_masks if item != []]
    table_token_type_ids = [torch.reshape(item, (-1, 512, 7)) for item in table_token_type_ids if item != []]

    batch_table_ids = torch.cat(table_input_ids)
    batch_table_attention_masks = torch.cat(table_attention_masks)
    batch_table_token_type_ids = torch.cat(table_token_type_ids)
    return batch_table_ids, batch_table_attention_masks, batch_table_token_type_ids

def connect_pooling_matrix(pooling_matrix):
    ### input: a list of matrix
    ### output: one matrix (分块)
    if not pooling_matrix:
        return torch.tensor([])
    if len(pooling_matrix) == 1:
        return pooling_matrix[0]
    left = pooling_matrix.pop(0)
    right = pooling_matrix[0]
    left_temp = torch.cat((left, torch.zeros(left.size()[0], right.size()[1])), dim = 1)
    right_temp = torch.cat((torch.zeros(right.size()[0], left.size()[1]), right), dim = 1) 
    pooling_matrix[0] = torch.cat((left_temp, right_temp))
    return connect_pooling_matrix(pooling_matrix)

class EvalMetric(object):
    def __init__(self):
        self.loss_meter = meter.AverageValueMeter()
        self.acc_meter = meter.AverageValueMeter()
        self.rec_meter = meter.AverageValueMeter()
        self.prec_meter = meter.AverageValueMeter()

    def meter_reset(self):
        self.loss_meter.reset()
        self.acc_meter.reset()
        self.rec_meter.reset()
        self.prec_meter.reset()

    def meter_add(self, acc, recall, prec, loss):
        self.loss_meter.add(loss)
        self.acc_meter.add(acc)
        self.rec_meter.add(recall)
        self.prec_meter.add(prec)

    def print_meter(self):
        print("loss:", self.loss_meter.value()[0])
        print("accuracy:", self.acc_meter.value()[0])
        print("recall:", self.rec_meter.value()[0])
        print("precision:", self.prec_meter.value()[0])
        print('\n')

def cal_matrix(pred_logits, golds, pred_epoch, gold_epoch, predict_threshold, args):
    assert len(pred_logits) == len(golds)
    golds = list(golds.cpu().detach().numpy())
    golds = [g for g in golds if g != -100]

    pred_scores = list(torch.exp(pred_logits[:len(golds)])[:, 1].cpu().detach().numpy())
    preds = [1 if item > predict_threshold else 0 for item in pred_scores]

    pred_epoch.extend(preds)
    gold_epoch.extend(golds)

    acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
    recall = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds) if sum(golds) else 1
    precision = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1
    return acc, recall, precision

def loss_and_metric(metric, loss, pred_logits, golds, pred_epoch, gold_epoch, predict_threshold, args):
    acc, recall, precision = cal_matrix(pred_logits, golds, pred_epoch, gold_epoch, predict_threshold, args)
    metric.meter_add(acc, recall, precision, loss.item())

def cell_distribution_analysis(split, args):
    ### analyze the number of rows and columns that has gold evidence in dataset
    in_path = args.input_path + '/{}.jsonl'.format(split)
    annotation_processor = AnnotationProcessor(in_path)
    col_results = defaultdict(int)
    row_results = defaultdict(int)
    five_square = 0
    cell_included = 0
    for anno in annotation_processor:
        row_set = set()
        col_set = set()
        evidence = anno.get_evidence(flat = True)
        for evi in evidence:
            if '_cell_' not in evi:
                continue
            id_temp = evi.split('_')
            col_id = id_temp[0] + id_temp[-3] + id_temp[-1]
            row_id = id_temp[0] + id_temp[-3] + id_temp[-2]
            col_set.add(col_id)
            row_set.add(row_id)
        col_results[len(col_set)] += 1
        row_results[len(row_set)] += 1
        if len(col_set) != 0:
            cell_included += 1
            if len(col_set) < 6 and len(row_set) < 6:
                five_square += 1
        
    print(col_results)
    print(row_results)
    print(cell_included)    
    print(five_square) 
    print(five_square/cell_included)       

def compute_metrics(preds, labels, target_names = ['NOTSEL', 'SELECT']):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    # class_rep = classification_report(labels, preds, target_names= ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES'], output_dict=False)
    # class_rep = classification_report(labels, preds, target_names=['NOTSEL', 'SELECT'], output_dict=False)

    class_rep = classification_report(labels, preds, target_names=target_names, output_dict=False)
    print(class_rep)
    print(acc, recall, precision, f1)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'class_rep': class_rep
    }

def detach(tensor):
    return tensor.detach().cpu().numpy()

def score_check(predicted_sents, predicted_cells, golds):
    ### return evidence group recall & accuracy & feverous score
    pred_evi = predicted_sents + predicted_cells
    evi_suffient = 0
    for evidence_group in golds:
        #actual_evi = [[e[0], e[1], e[2]] for e in evidence_group]
        if all([evi in pred_evi for evi in evidence_group]):
            evi_suffient = 1
            break
    return evi_suffient 

def cell_mask_generation(row_num, col_num):
    ### row/col nums of each table [x1,x2,x3]
    ### probability matrix: R(row_num, col_num)
    ### generate a mask to select solid cells in this matrix
    mask = []
    for x, y in zip(row_num, col_num):
        mask.append(torch.ones((x, y)))
    return connect_pooling_matrix(mask)

def get_node_match_matrix(sent_num, row_num, col_num):
    ### sent_num: a list of sent num in each data
    ### row_num: a list of row num in each data
    ### col_num: a list of col num in each data
    ### the graph ids are [sent_ids data1, row_ids data1, col_ids data1, sent_ids data2, row_ids data2, col_ids data2, ...
    ### the input ids are [sent_ids data1, sent_ids data2, ... row_ids data1, row_ids data2, ... col_ids data1, col_ids data2, ...]
    ### we generate a matrix to match the position of sent_ids in the graph and the input
    ### the matrix is (sum(sent_num) + sum(row_num) + sum(col_num), sum(sent_num) + sum(row_num) + sum(col_num))
    ### Embeddings dim [num_nodes, 768]
    ### matched_embeddings = torch.matmul(matrix, embeddings)

    matrix = torch.zeros((sum(sent_num) + sum(row_num) + sum(col_num), sum(sent_num) + sum(row_num) + sum(col_num)))
    sent_start, row_start, col_start = 0, 0, 0
    src_sent_start, src_row_start, src_col_start = 0, sum(sent_num), sum(sent_num + row_num)
    for i in range(len(sent_num)):
        src_sent_start += sent_num[i-1] if i > 0 else 0
        for j in range(sent_num[i]):
            matrix[src_sent_start + j, sent_start + j] = 1
        sent_start += sent_num[i] + row_num[i] + col_num[i]
    for i in range(len(row_num)):
        src_row_start += row_num[i-1] if i > 0 else 0
        row_start += sent_num[i] 
        for j in range(row_num[i]):
            matrix[src_row_start + j, row_start + j] = 1
        row_start += row_num[i] + col_num[i]
    for i in range(len(col_num)):
        src_col_start += col_num[i-1] if i > 0 else 0
        col_start += sent_num[i] + row_num[i]
        for j in range(col_num[i]):
            matrix[src_col_start + j, col_start + j] = 1
        col_start += col_num[i]
    return matrix


def linearize_cell_id(table_ids_2D):
    ### input: a list of tables as in [[[cell_id1, 2,3], []], table] list of cell ids
    ### output: list of cell ids
    return [id for table in table_ids_2D for row in table for id in row]

def save_entry(entry, path, file_name):
    save_path = path + '/' + file_name
    print('Predicted Evidence and Verdict are save to ' + save_path)
    with jsonlines.open(save_path, 'w') as writer:
        for evi in entry:
            writer.write(evi)

def get_sent_cell_golds(gold_evidence):
    gold_sents, gold_cells = set(), set()
    for evi in gold_evidence:
        if '_sentence_' in evi:
            gold_sents.add(evi)
        elif '_cell_' in evi:
            gold_cells.add(evi)
    return gold_sents, gold_cells
        
if __name__ == '__main__':
    test_case = [[[1,2], [3,4]],[[5,6,7],[2,3,4,5]]]
    print(linearize_cell_id(test_case))
