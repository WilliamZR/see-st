import pandas as pd
from collections import defaultdict
import sys
sys.path.append('/home/wuzr/feverous/src')
from my_utils.common_utils import load_pkl_data, save_pkl_data
import pickle
import numpy as np
import dgl 
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
from graph_parser import get_parser
from util import connect_pooling_matrix
from baseline.drqa.tokenizers.spacy_tokenizer import SpacyTokenizer
import jsonlines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, default = 'data')
    parser.add_argument('--input_file', type = str)
    parser.add_argument('--split', type = str, default='dev')
    args = parser.parse_args()
    print(args)

    predicted_sentences_dict = {}
    with jsonlines.open('{0}/{1}.sentences.roberta.p5.s5.jsonl'.format(args.input_path, args.split), 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            predicted_sentences_dict[line['id']] = line['predicted_sentences']  
    fusion_dict = {}
    with jsonlines.open('{0}/{1}'.format(args.input_path, args.input_file), 'r') as f:
        for i, line in enumerate(f):
            fusion_dict[int(line['id'])] = line['predicted_evidence']


    with jsonlines.open('{0}/{1}.jsonl'.format(args.input_path, args.split), 'r') as f: 
    ### Output keys :, 'id', 'claim', 'evidence', 'label', 'predicted_evidence'
        with jsonlines.open('{0}/{1}.scorecheck.jsonl'.format(args.input_path, args.split), 'w') as writer:
            for i, line in enumerate(f):
                if i == 0:
                    writer.write({'header':''})
                    continue
                id = line['id']
                output = {'id': id,
                          'claim': line['claim'],
                          'evidence': line['evidence'],
                          'label': line['label'] if 'label' in line.keys() else 'SUPPORTS',
                          'predicted_evidence': fusion_dict[id] if id in fusion_dict else predicted_sentences_dict[id]}
                writer.write(output)
            if 'label' not in line.keys():
                print('Gold Label Not Found...Use SUPPORTS for all instance... \n If you are using test set, please ignore this warning.')
    print('Done')