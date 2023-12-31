import sys

import argparse
import json
from tqdm import tqdm
from utils.annotation_processor import AnnotationProcessor, EvidenceType
import unicodedata
from cleantext import clean
from urllib.parse import unquote
from utils.prepare_model_input import get_wikipage_by_id, init_db

import os
import pandas as pd
# os.chdir("../../../")




def clean_title(text):
    text = unquote(text)
    text = unicodedata.normalize('NFD', text)
    text = clean(text.strip(), fix_unicode=True,  # fix various unicode errors
                 to_ascii=False,  # transliterate to closest ASCII representation
                 lower=False,  # lowercase text
                 no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
                 no_urls=True,  # replace all URLs with a special token
                 no_emails=False,  # replace all email addresses with a special token
                 no_phone_numbers=False,  # replace all phone numbers with a special token
                 no_numbers=False,  # replace all numbers with a special token
                 no_digits=False,  # replace all digits with a special token
                 no_currency_symbols=False,  # replace all currency symbols with a special token
                 no_punct=False,  # remove punctuations
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="0",
                 replace_with_currency_symbol="<CUR>",
                 lang="en"  # set to 'de' for German special handling
                 )
    return text


def extract_tables_from_evidence(evidence_list):
    new_evidence = []
    for ev in evidence_list:
        if '_cell_' in ev:
            table_id = ev.split('_')[0] + '_table_' + ev.split('_')[2]
            if table_id not in new_evidence:
                new_evidence.append(table_id)
        elif '_header_cell' in ev:
            table_id = ev.split('_')[0] + '_table_' + ev.split('_')[3]
            if table_id not in new_evidence:
                new_evidence.append(table_id)
        elif '_item_' not in ev and '_caption_' not in ev:
            new_evidence.append(ev)
    return new_evidence


def average(list):
    print(len(list))
    return float(sum(list) / len(list))


def evidence_coverage(args):
    print('Evidence coverage...')
    log = 0
    coverage = []
    coverage_all = []
    annotation_processor = AnnotationProcessor('data/{}.jsonl'.format(args.split))
    if args.all == 0:
        annotation_by_id = {i: el for i, el in enumerate(annotation_processor) if
                            el.has_evidence() and el.get_evidence_type(flat=True) == EvidenceType.SENTENCE}
    else:
        annotation_by_id = {i: el for i, el in enumerate(annotation_processor) if el.has_evidence()}

    gold_cell_num = []
    pred_cell_num = []
    #with open('/home/wuzr/feverous/data/{}.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl'.format(args.split), 'r') as f:
    input_path = '/home/wuzr/see-st/data/dev.scorecheck.jsonl'
    with open(input_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            js = json.loads(line)
            id = idx - 1 # js['id']
            if id not in annotation_by_id:
                continue
            anno = annotation_by_id[id]
            docs_gold = list(set([t for t in anno.get_evidence(flat=True) if '_cell_' in t]))

            if len(docs_gold) == 0:
                continue

            docs_predicted = [t for t in js['predicted_evidence'] if '_cell_' in t]
            gold_cell_num.append(len(docs_gold))
            pred_cell_num.append(len(docs_predicted))

            if anno.get_verdict() in ['SUPPORTS', 'REFUTES']:
                coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
                coverage.append(coverage_ele)
                coverage_all.append(coverage_ele)
                
            else:
                coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
                coverage_all.append(coverage_ele)

    print("average gold cell num:", average(gold_cell_num))
    print("average pred cell num:", average(pred_cell_num))

    print(average(coverage))
    print(average(coverage_all))

def evidence_coverage_path(pred_path, gold_path):
    print('Evidence coverage...')
    coverage = []
    coverage_all = []
    annotation_processor = AnnotationProcessor(gold_path)
    annotation_by_id = {i: el for i, el in enumerate(annotation_processor) if el.has_evidence()}

    with open(pred_path, "r") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            js = json.loads(line)
            id = idx - 1  # js['id']
            if id not in annotation_by_id:
                continue
            anno = annotation_by_id[id]
            docs_gold = list(set([t for t in anno.get_evidence(flat=True) if '_cell_' in t]))

            if len(docs_gold) == 0:
                continue

            docs_predicted = [t for t in js['predicted_evidence'] if '_cell_' in t]

            if anno.get_verdict() in ['SUPPORTS', 'REFUTES']:
                coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
                coverage.append(coverage_ele)
                coverage_all.append(coverage_ele)
                

            else:
                coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
                coverage_all.append(coverage_ele)
    print(average(coverage))
    print(average(coverage_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--max_page', type=int, default=5)
    parser.add_argument('--max_sent', type=int, default=5)
    parser.add_argument('--max_tabs', type=int, default=3)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--all', type=int, default=1)
    parser.add_argument('--wiki_path',default='data/feverous_wikiv1.db',type=str)
    args = parser.parse_args()
    init_db(args.wiki_path)
    evidence_coverage(args)

# PYTHONPATH=src python src/baseline/retriever/eval_cell_retriever.py --split dev --max_page 5 --max_sent 5 --max_tabs 3