### eval_rc_retriever.py
### Wu Zirui
### 20221031
### 
import sys

import argparse
from my_method.table_extraction.rc_retriever_parser import get_parser
from train_rc_retriever import val, compute_table_retrieval, collate_fn, get_gold_table_evidence
import torch
from prepare_rc_data import RC_Generator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import jsonlines
from rc_model import RC_MLP_Retriever


if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)
    for eval_split in ['train', 'dev', 'test']:
        tokenizer = AutoTokenizer.from_pretrained(args.tapas_path)
        gold_evidence_by_id = get_gold_table_evidence(eval_split, args.input_path)
        model = RC_MLP_Retriever(args)
        ckpt_meta = model.load(args.model_load_path)
        assert ckpt_meta['select_criterion'] == args.select_criterion
        model.to(args.device)
        model.eval()

        valid_data = RC_Generator(args.input_path, tokenizer, eval_split, args)
        valid_data.get_data_for_epoch(train = False)
        valid_data_loader = DataLoader(valid_data, batch_size = 1, shuffle = False, collate_fn = collate_fn)

        table_scores_by_id = val(model, valid_data_loader, valid_data, args)
        recall, predicted_tables_by_id = compute_table_retrieval(table_scores_by_id, gold_evidence_by_id, args)
        print(eval_split)
        print('Table Retrieval:')
        print(recall)

        with jsonlines.open(args.input_path + '/' + eval_split + '.rc.p5.t{0}.jsonl'.format(args.max_tabs), 'w') as writer:
            for id in predicted_tables_by_id:
                writer.write({'id':id,                                  
                              'predicted_tables':predicted_tables_by_id[id]})

