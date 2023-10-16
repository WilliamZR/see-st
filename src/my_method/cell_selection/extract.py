import sys
import os
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
from base_templates import BasePreprocessor
from my_utils.pytorch_common_utils import set_seed, get_optimizer 
from my_utils.task_metric import compute_metrics
from my_utils.common_utils import average
from my_utils.torch_model_utils import print_grad
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from graph_parser import get_parser
from my_utils.pytorch_common_utils import set_seed
from utils.annotation_processor import AnnotationProcessor
from util import * 
from graph_generator import RowColSentFusionGraphGenerator
from graph_model import FusionGraphModel
import dgl
from train_graph_model import collate_fn

@torch.no_grad()
def main():
    args = get_parser().parse_args()
    print(args)
    set_seed(args.seed)

    roberta_tokenizer = AutoTokenizer.from_pretrained(args.roberta_path)
    tapas_tokenizer = AutoTokenizer.from_pretrained(args.tapas_path)

    for split in ['dev', 'test']:
        data = RowColSentFusionGraphGenerator(roberta_tokenizer, tapas_tokenizer, split, args)
        dataloader = DataLoader(data, batch_size = 1, shuffle = False, collate_fn = collate_fn)

        if split != 'test':
           annotations = AnnotationProcessor('{0}/{1}.jsonl'.format(args.input_path, split))
           gold_evidence = {anno.get_id(): anno.get_evidence() for anno in annotations}

        model = FusionGraphModel(args)
        ckpt_meta = model.load(args.model_load_path)
        model.to(args.device)
        model.eval()
        entry = []
        sent_recall_all = []
        cell_recall_all = []
        recall_all, acc_all, fever_score_all = 0, 0, 0

        label2verdict = {1: 'SUPPORTS', 0: 'NOT ENOUGH INFO', 2: 'REFUTES'}
        for ii, batch in tqdm(enumerate(dataloader)):
            ### compute logits
            if ii == 10:
                break
            batch = [item.to(args.device) for item in batch]   
            input_data = batch[:-5]
            verdict_label = batch[-1]
            res = model(input_data)

            sent_pred_logits, row_pred_logits, col_pred_logits, cell_pred_logits, verdict_pred_logits = res
            sent_pred_logits, cell_pred_logits, verdict_pred_logits = map(detach, [sent_pred_logits, cell_pred_logits, verdict_pred_logits])

            ### select evidence
            sent_ids = data.raw_data[ii]['sent_row_col_id']['sent']
            table_ids_2D = data.raw_data[ii]['table_ids_2D']
            cell_ids = linearize_cell_id(table_ids_2D)
            evidence_id = data.raw_data[ii]['id']
            #### Sentences threshold
            sent_score = list(zip(sent_ids, sent_pred_logits[:, -1]))
            predicted_sents = [item[0] for item in sent_score if np.exp(item[1]) > args.sent_threshold]
            #### Cells top 25
            #### 在建图的时候，不同的cell可能对应同一个id，需要在这里进行一个去重
            predicted_cells = []
            if cell_pred_logits.size != 1:
                cell_score = list(zip(cell_ids, cell_pred_logits[:, -1]))
                cell_score.sort(key = lambda x : float(x[1]), reverse = True)
                sorted_cells = list(list(zip(*cell_score))[0])
                for cell in sorted_cells:
                    if cell not in predicted_cells and len(predicted_cells) < 25:
                        predicted_cells.append(cell)
                    if len(predicted_cells) == 25:
                        break

            #### Verdict
            verdict = np.argmax(verdict_pred_logits)
            
            ### Compute F1 and Feverous Score metrics
            golds = []
            if split != 'test':
                golds = gold_evidence[evidence_id]
                flatten_golds = [evi for group in golds for evi in group]
                gold_sents, gold_cells = get_sent_cell_golds(flatten_golds)
                cover_sents, cover_cells = 0, 0
                if gold_sents:
                    cover_sents = len(set(gold_sents) & set(predicted_sents))            
                    sent_recall_all.append(cover_sents / len(gold_sents))
                if gold_cells:
                    cover_cells = len(set(gold_cells) & set(predicted_cells))
                    cell_recall_all.append(cover_cells / len(gold_cells))
                recall, acc, feverous_score = score_check(predicted_sents, predicted_cells, golds, verdict, verdict_label)
                recall_all += recall / len(dataloader)
                acc_all += acc / len(dataloader)
                fever_score_all += feverous_score / len(dataloader)
            entry.append({
                'id': str(evidence_id), 
                'evidence': golds,
                'predicted_evidence': predicted_sents + predicted_cells, 
                'predicted_label': label2verdict[verdict],
                'label': label2verdict[int(detach(verdict_label))]})
        print(split)
        if split != 'test':
            print('Sentence Recall {:.5f}'.format(average(sent_recall_all)))
            print('Cell Recall: {:.5f}'.format(average(cell_recall_all)))
            print('Feverous Score {:.5f}'.format(fever_score_all))
            print('Accuracy: {:.5f}'.format(acc_all))
            print('Evidence Set Recall: {:.5f}'.format(recall_all))

        save_entry(entry, args.input_path,split+'.'+ args.entry_save_file)
if __name__ == '__main__':
    main()
