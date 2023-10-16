import pandas as pd
from collections import defaultdict
import sys
from my_utils.common_utils import load_pkl_data, save_pkl_data
import pickle
import numpy as np
import dgl 
from multiprocessing import Pool
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel, TapasTokenizer
import math
import argparse
from tqdm import tqdm
import os
import re
import torch
import json
from utils.annotation_processor import AnnotationProcessor, EvidenceType
from utils.wiki_page import WikiPage, get_wikipage_by_id,WikiTable
from database.feverous_db import FeverousDB
from graph_parser import get_parser
from util import average_of_matrix
from util import connect_pooling_matrix
from baseline.drqa.tokenizers.spacy_tokenizer import SpacyTokenizer
from collections import Counter
import jsonlines
import unicodedata
from multiprocessing.pool import ThreadPool
from tqdm.contrib.concurrent import process_map, thread_map
TOKENIZER = SpacyTokenizer(annotators=set(['ner']))

def clean_hyperlink_brakets(sentence):
    hyperlink = re.findall('\[\[.*?\|', sentence)
    hyperlink = [item[2:-1] for item in hyperlink]
    sentence = re.sub('\[\[.*?\|', '', sentence)
    sentence = re.sub('\]\]', '', sentence)
    return hyperlink, sentence

class RowColSentFusionGraphGenerator(torch.utils.data.Dataset):
    def __init__(self, roberta_tokenizer, tapas_tokenizer, data_type, args):
        super().__init__()
        self.roberta_tokenizer = roberta_tokenizer
        self.tapas_tokenizer = tapas_tokenizer
        self.data_type = data_type
        self.args = args
        self.raw_data = self.get_raw_data(new_cache = args.new_cache, cache_name=args.cache_name)
        self.clean_raw_data()
        self.load_edges(data_type,args.rebuild_edges, cache_name = args.cache_name)
        self.verdict2label = {
        "NOT ENOUGH INFO": 0,
        "SUPPORTS": 1,
        "REFUTES": 2,
    }   

    def clean_raw_data(self):
        def keep_instance(instance):
            if (1 not in instance['sent_labels'] and 1 not in instance['cell_labels']) and self.data_type == 'train':
                return False
            if not instance['sentences']:
                return False
            return True
        self.raw_data = [instance for instance in self.raw_data if keep_instance(instance)]
        print(len(self.raw_data))

    def build_cache(self, data_split, args, input_name = 'roberta_sent.rc_table.not_precomputed.p5.s5.t5', cache_name = 'rc_fusion_graph'):        
        cache_file = 'data/{0}.{1}.jsonl'.format(data_split, cache_name)
        assert not os.path.exists(cache_file) or args.new_cache
        
        anno_file = 'data/{0}.{1}.jsonl'.format(data_split, input_name)
        print('Loading predicted evidence from ' + anno_file)
        anno_processor = AnnotationProcessor(anno_file)
        annotations = [anno for anno in anno_processor]
        if args.cache_workers == 1:
            output = self.graph_worker(annotations)
        else:
            output = process_map(self.build_graph, annotations, max_workers = args.cache_workers, chunksize = 5)
        save_pkl_data(output, cache_file)
        print('Data saved at ' + cache_file)

    def graph_worker(self, annotations):
        output = []
        for i, anno in tqdm(enumerate(annotations)):    
    ### claim for roberta and tapas
    ### sentence for roberta
    ### table for tapas
    ### table_ids_2d for pooling and selecting cells
    ### edges
    ### sentence_id
    ### sentence labels
    ### col labels
    ### row labels
    ### cell labels
    ### verdict label
    ### the ids are collected. These can be used to construct graphs afterwards
            #instance = self.build_graph(anno, nlp, args)
            instance = self.build_graph(anno)
            if 1 not in instance['sent_labels'] and 1 not in instance['cell_labels']:
                if self.data_type == 'train':
                    continue
            output.append(instance)
        return output
    
    def build_edges_cache(self, data_split, pattern, cache_name = 'edges_cache'):
        cache_file = 'data/' + '{0}.{1}.{2}.jsonl'.format(data_split, self.args.edge_pattern, cache_name)
        self.add_all_instance_edges(pattern = pattern)
        edges = {instance['id']: instance['edges'] for instance in self.raw_data}
        save_pkl_data(edges, cache_file)

    def load_edges(self, data_split, rebuild = False, cache_name = 'edges_cache'):
        cache_file = 'data/' + '{0}.{1}.{2}.jsonl'.format(data_split, self.args.edge_pattern, cache_name)
        if not os.path.exists(cache_file) or rebuild:
            self.build_edges_cache(data_split, self.args.edge_pattern, cache_name=cache_name)
        edges_cache = load_pkl_data(cache_file)
        for i in tqdm(range(len(self.raw_data)), desc = 'Loading Edges From ' + cache_file):
            self.raw_data[i]['edges'] = edges_cache[self.raw_data[i]['id']]

        

    def build_graph(self, anno):
        ### input: annotation instance
        ### output:instance with graph and labels
        args = self.args
        args.db = FeverousDB(args.wiki_path)
        id = anno.id
        claim = anno.claim
        verdict_label = anno.get_verdict()

        sentence_ids = []
        table_ids = set()
        cell_ids = []
        table_content = []
        table_ids_2D = []
        cell_id_by_table = defaultdict(list)
        if self.data_type == 'test':
            gold_evidence = []
        else:
            gold_evidence = anno.get_evidence(flat = True)

        for evi in anno.predicted_evidence:
            if '_sentence_' in evi:
                sentence_ids.append(evi)
            elif '_table_' in evi:
                table_ids.add(evi)
            elif '_cell_' in evi:
                table_id = evi.split('_')[0] + '_table_' + evi.split('_')[-3]
                if table_id in table_ids:
                    cell_id_by_table[table_id].append(evi)

        sentences = []
        entity_pool = defaultdict(set)
        hyperlink_pool = defaultdict(set)
        sentence_ids = list(set(sentence_ids))
        for evi in sentence_ids:
            page_id = evi.split('_')[0]
            sent_id = '_'.join(evi.split('_')[1:])
            page_json = args.db.get_doc_json(page_id)
            curr_page = WikiPage(page_id, page_json)
            if curr_page is None:
                continue
            hyperlink, sentence = clean_hyperlink_brakets(curr_page.get_element_by_id(sent_id).content)
            for item in hyperlink:
                hyperlink_pool[item].add(evi)
            sentences.append(sentence)
        if sentences:
            for i, sent in enumerate(sentences):
                entity_group = TOKENIZER.tokenize(sent).entity_groups()
                if entity_group:
                    for pair in entity_group:
                        entity_pool[pair[0]].add(sentence_ids[i])
        

        col_labels = []
        row_labels = []
        for evi in table_ids:
            page_id = evi.split('_')[0]
            table_id = '_'.join(evi.split('_')[1:])
            page_json = args.db.get_doc_json(page_id)
            curr_page = WikiPage(page_id, page_json)
            if curr_page is None:
                continue
            curr_table = curr_page.get_element_by_id(table_id)
            curr_id_2D, curr_content,curr_row_labels, curr_col_labels, curr_cell_ids = self.prepare_table_with_entity(curr_table, gold_evidence)
            cell_ids.append(curr_cell_ids)

            for i, row in enumerate(curr_content):
                for j, content in enumerate(row):
                    hyperlink, content = clean_hyperlink_brakets(content)
                    for item in hyperlink:
                        hyperlink_pool[item].add(curr_id_2D[i][j])
                    curr_content[i][j] = content
            for j, curr_row_content in enumerate(curr_content):
                for i, sent in enumerate(curr_row_content):
                    entity_group = TOKENIZER.tokenize(sent).entity_groups()
                    if entity_group:
                        for pair in entity_group:
                            entity_pool[pair[0]].add((page_id + '_' + table_id, j, i))
  
            table_content.append(curr_content)
            table_ids_2D.append(curr_id_2D)
            col_labels.append(curr_col_labels)        
            row_labels.append(curr_row_labels)

        sent_row_col_edges, sent_row_col_ids = self.build_edges_based_on_entity(sentence_ids, entity_pool, hyperlink_pool, table_ids, table_ids_2D)

        sentence_labels = [1 if evi in gold_evidence else 0 for evi in sentence_ids]
        cell_labels = [1 if evi in gold_evidence else 0  for item in cell_ids for evi in item]
        col_labels = [evi_label for item in col_labels for evi_label in item]
        row_labels = [evi_label for item in row_labels for evi_label in item]
        output = {
            'id': id,
            'claim': claim,
            'sentences': sentences,
            'tables': table_content,
            'table_ids_2D': table_ids_2D,
            'sent_row_col_id': sent_row_col_ids,
            'sent_labels': sentence_labels,
            'row_labels': row_labels,
            'col_labels': col_labels,
            'cell_labels': cell_labels,
            'verdict_label': verdict_label,
            'edges': sent_row_col_edges
        }

        return output

    def get_raw_data(self, new_cache = False, cache_name = 'rc_fusion_graph'):
        cache_file = 'data/{0}.{1}.jsonl'.format(self.data_type, cache_name)
        print('cache_path:' + cache_file)
        if new_cache or not os.path.exists(cache_file):
            self.build_cache(self.data_type, self.args, self.args.input_name, cache_name)
        print('Cache File:' + cache_file)
        return load_pkl_data(cache_file)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        '''
        output = {
            'id': id,
            'claim': claim,
            'sentences': sentences,
            'tables': table_content,
            'table_ids_2D': table_ids_2D,
            'sent_row_col_id': sent_row_col_ids, ## dict {sent:[], col:[], row:[]}
            'sent_labels': sentence_labels,
            'row_labels': row_labels,
            'col_labels': col_labels,
            'cell_labels': cell_labels,
            'verdict_label': verdict_label,
            'edges': sent_row_col_edges (only contain edges for entity and hyperlink)
        }
        '''
        instance = self.raw_data[idx]
        claim = instance['claim']
        sentences = instance['sentences']
        tables = instance['tables']
        table_ids_2D = instance['table_ids_2D']
        #ids_dict = instance['sent_row_col_id']
        #sent_row_col_ids = ids_dict['sent'] + ids_dict['row'] + ids_dict['col']
        sent_labels = instance['sent_labels']
        row_labels = instance['row_labels']
        col_labels = instance['col_labels']
        cell_labels = instance['cell_labels']
        verdict_label = instance['verdict_label']
        edges = instance['edges']
        #edges = self.add_edges(edges, sent_row_col_ids, pattern=self.edge_pattern)
        claims = [claim] * len(sentences)

        sentence_input = self.roberta_tokenizer(claims, sentences, padding = 'max_length', max_length = 512, truncation = True, return_tensors = 'pt')
        
        table_input_ids = []
        table_attention_masks = []
        table_token_type_ids = []
        row_pooling_matrix = []
        col_pooling_matrix = []
        row_nums = []
        col_nums = []

        for i in range(len(tables)):
            temp_input, temp_mask, temp_token_type, temp_row_pooling_matrix, temp_col_pooling_matrix = self.tokenize_claim_and_inputs(claim, tables[i], table_ids_2D[i], self.tapas_tokenizer)
            table_input_ids.append(temp_input)
            table_attention_masks.append(temp_mask)
            table_token_type_ids.append(temp_token_type)
            row_pooling_matrix.append(temp_row_pooling_matrix)
            col_pooling_matrix.append(temp_col_pooling_matrix)
            row_nums.append(temp_row_pooling_matrix.size()[0])
            col_nums.append(temp_col_pooling_matrix.size()[0])
        if not tables:
            table_input_ids, table_attention_masks, table_token_type_ids = [],[],[]
            row_pooling_matrix, col_pooling_matrix = [], []
        else:
            table_input_ids, table_attention_masks, table_token_type_ids = map(torch.stack, [table_input_ids, table_attention_masks, table_token_type_ids])
            row_pooling_matrix, col_pooling_matrix = map(connect_pooling_matrix, [row_pooling_matrix, col_pooling_matrix])

        graph = dgl.graph(list(edges))
        graph = dgl.add_self_loop(graph)
        if row_pooling_matrix != []:
            graph.ndata['t'] = torch.tensor([0] * len(sentences) + [1] * row_pooling_matrix.size()[0] + [2] * col_pooling_matrix.size()[0])
        else:
            graph.ndata['t'] = torch.tensor([0] * len(sentences))
        return sentence_input['input_ids'], sentence_input['attention_mask']\
            , table_input_ids, table_attention_masks, table_token_type_ids\
            , row_pooling_matrix, col_pooling_matrix, len(sent_labels), row_nums, col_nums\
            , graph, sent_labels, row_labels, col_labels, cell_labels

    def tokenize_claim_and_inputs(self, claim, table, table_ids, tokenizer):
        table =pd.DataFrame(table, columns=table[0], dtype = str).fillna('')
        tokenized_inputs = tokenizer(table = table, queries = claim, padding = 'max_length', truncation = True, return_tensors = 'pt')
        row_pooling_matrix, col_pooling_matrix = self.get_pooling_matrix(tokenized_inputs['token_type_ids'], table_ids)
        return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], tokenized_inputs['token_type_ids'], row_pooling_matrix, col_pooling_matrix
        
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
    
    def add_edges(self, edges, sent_row_col_ids, pattern = 'full_connection'):
        def edge_connection(evi_group_i, evi_group_j, edges, id_seq):
            for evi_i in evi_group_i:
                for evi_j in evi_group_j:
                    edges.add((id_seq.index(evi_i), id_seq.index(evi_j)))
            return edges
        id_sequence = sent_row_col_ids['sent'] + sent_row_col_ids['row'] + sent_row_col_ids['col']
        if pattern == 'full_connection':
            for i in range(len(id_sequence)):
                for j in range(len(id_sequence)):
                    edges.add((i, j))
        elif pattern == 'page_related':
            page_pool = defaultdict(list)
            for evi in id_sequence:
                page = evi.split('_')[0]
                page_pool[page].append(evi)
            for page in page_pool:
                edges = edge_connection(page_pool[page], page_pool[page], edges, id_sequence)
        elif pattern == 'page_and_table':
            sent_page_pool = defaultdict(list)
            row_col_page_pool = defaultdict(list)
            table_pool = defaultdict(list)
            for evi in id_sequence:
                evi_split = evi.split('_')
                page = evi_split[0]
                if evi_split[1] == 'sentence':
                    sent_page_pool[page].append(evi)
                else:
                    row_col_page_pool[page].append(evi)
                    table_id = '_'.join(evi_split[:3])
                    table_pool[table_id].append(evi)
            for page in sent_page_pool:
                edges = edge_connection(sent_page_pool[page], sent_page_pool[page], edges, id_sequence)
                edges = edge_connection(row_col_page_pool[page], sent_page_pool[page], edges, id_sequence)
            for table_id in table_pool:
                edges = edge_connection(table_pool[table_id], table_pool[table_id], edges, id_sequence)
        elif pattern == 'entity_only':
            for i in range(len(id_sequence)):
                edges.add((i, i))

        else:
            raise NotImplementedError
        return edges

    def add_all_instance_edges(self, pattern):
        for instance in tqdm(self.raw_data, desc = 'Adding edges as in {} pattern'.format(pattern)):
            instance['edges'] = self.add_edges(instance['edges'], instance['sent_row_col_id'], pattern = pattern)

    def prepare_table_with_entity(self, curr_tab, gold_evidence):
        page = curr_tab.page
        table_content_2D = []
        table_ids_2D = []
        col_labels = set()
        row_labels = set()
        output_candidates = []
        
        for i, row in enumerate(curr_tab.rows):
            if i == 256:
                break
            row_id = []
            row_flat = []
            for j, cell in enumerate(row.row):
                if j == 128:
                    break
                curr_id = page + '_' + cell.get_id()
                if curr_id in gold_evidence:
                    col_labels.add(j)
                    row_labels.add(i)
                row_id.append(curr_id)
                row_flat.append(str(cell))
                output_candidates.append(curr_id)

            table_ids_2D.append(row_id)
            table_content_2D.append(row_flat)

        col_labels = [1 if i in col_labels else 0 for i in range(len(table_ids_2D[0]))]
        row_labels = [1 if i in row_labels else 0 for i in range(len(table_ids_2D))]

        return table_ids_2D, table_content_2D, row_labels ,col_labels, output_candidates

    def build_edges_based_on_entity(self, sentence_ids,entity_pool, hyperlink_pool, table_ids, table_content_2D):
        def generate_id_sequence(table_ids, table_content_2D):
            col_id_sequence = []
            row_id_sequence = []
            for i, curr_tab_id in enumerate(table_ids):
                for j in range(len(table_content_2D[i][0])):
                    col_id_sequence.append(curr_tab_id + '_col_' + str(j))
                for k in range(len(table_content_2D[i])):
                    row_id_sequence.append(curr_tab_id + '_row_' + str(k))
            return row_id_sequence, col_id_sequence
        
        def connect_edges_in_pool(evidence_pool, evi_ids):
            edges = set()
            pool_reshaped_format = set()
            for evi in evidence_pool:
                if '_sentence_' in evi:
                    pool_reshaped_format.add(evi)
                elif '_table' in evi[0]:##（table_id, row_num, col_num）
                    tab_id = evi[0]
                    col_id = tab_id + '_col_' + str(evi[2])
                    row_id = tab_id + '_row_' + str(evi[1])
                    pool_reshaped_format.add(col_id)
                    pool_reshaped_format.add(row_id) 
            for i in pool_reshaped_format:
                for j in pool_reshaped_format:
                    edges.add((evi_ids.index(i), evi_ids.index(j)))
            return edges


        edges = set()
        row_ids, col_ids = generate_id_sequence(table_ids, table_content_2D)
        sent_row_col_ids = sentence_ids + row_ids + col_ids

        for key in entity_pool.keys():
            curr_edges = connect_edges_in_pool(entity_pool[key], sent_row_col_ids)
            edges = edges.union(curr_edges)
        for key in hyperlink_pool.keys():
            curr_edges = connect_edges_in_pool(hyperlink_pool, sent_row_col_ids)
            edges = edges.union(curr_edges)

        return edges, {'sent': sentence_ids, 'row': row_ids, 'col': col_ids}

        
if __name__ == '__main__':
    args = get_parser().parse_args()
    roberta_tokenizer = AutoTokenizer.from_pretrained(args.roberta_path)
    tapas_tokenizer = AutoTokenizer.from_pretrained(args.tapas_path)
    for split in ['dev','train','test']:
        data = RowColSentFusionGraphGenerator(roberta_tokenizer, tapas_tokenizer, split, args)
