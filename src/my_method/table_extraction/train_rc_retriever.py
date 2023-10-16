import argparse
import sys

import os
import torch
from tqdm import tqdm

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
from rc_retriever_parser import get_parser
from my_utils.pytorch_common_utils import set_seed
from prepare_rc_data import RC_Generator
from rc_model import RC_MLP_Retriever
from torchnet import meter
from utils.annotation_processor import AnnotationProcessor
from util import EvalMetric, cal_matrix, connect_pooling_matrix, reshape_input, reshape_labels

def collate_fn(batch):
    pos_ids, pos_masks, pos_token_ids\
            , neg_ids, neg_masks, neg_token_ids\
            , pos_row_pooling_matrix, pos_col_pooling_matrix\
            , neg_row_pooling_matrix, neg_col_pooling_matrix\
            , pos_row_labels, pos_col_labels\
            , neg_row_labels, neg_col_labels = map(list, zip(*batch))

    batch_ids, batch_masks, batch_token_ids = reshape_input(pos_ids, pos_masks, pos_token_ids)
    row_pooling_matrix, col_pooling_matrix = map(connect_pooling_matrix, [pos_row_pooling_matrix, pos_col_pooling_matrix])
    row_labels, col_labels = map(reshape_labels, [pos_row_labels, pos_col_labels])

    if neg_col_labels[0] != None:
        batch_neg_ids, batch_neg_masks, batch_neg_token_ids = reshape_input(neg_ids, neg_masks, neg_token_ids)
        neg_row_pooling_matrix, neg_col_pooling_matrix = map(connect_pooling_matrix, [ neg_row_pooling_matrix, neg_col_pooling_matrix])
        neg_row_labels, neg_col_labels = map(reshape_labels, [neg_row_labels, neg_col_labels])

        row_labels += neg_row_labels
        col_labels += neg_col_labels

        batch_ids = torch.cat([batch_ids, batch_neg_ids])
        batch_masks = torch.cat([batch_masks, batch_neg_masks])
        batch_token_ids = torch.cat([batch_token_ids, batch_neg_token_ids])

        row_pooling_matrix = connect_pooling_matrix([row_pooling_matrix, neg_row_pooling_matrix])
        col_pooling_matrix = connect_pooling_matrix([col_pooling_matrix, neg_col_pooling_matrix])
        
        
    row_labels = torch.tensor(row_labels)
    col_labels = torch.tensor(col_labels)

    return batch_ids, batch_masks, batch_token_ids, row_pooling_matrix, col_pooling_matrix, row_labels, col_labels

@torch.no_grad()
def val(model, data_loader, dataset, args):
    model.eval()
    table_score_by_id = defaultdict(list)
    for i, batch in tqdm(enumerate(data_loader)):
        
        batch = [item.to(args.device) for item in batch]
        input_data = batch[:-2]
        res = model(input_data, test_mode = True)
        col_res, row_res, table_res, _ = res

        if args.select_criterion == 'row+col':
            score = torch.max(torch.exp(col_res[:, 1])) + torch.max(torch.exp(row_res[:, 1]))
        elif args.select_criterion == 'row*col':
            score = torch.mul(torch.max(torch.exp(col_res[:, 1])), torch.max(torch.exp(row_res[:, 1])))
        elif args.select_criterion == 'row':
            score = torch.max(torch.exp(row_res[:, 1]))
        elif args.select_criterion == 'col':
            score = torch.max(torch.exp(col_res[:, 1]))
        else:
            raise NotImplementedError
        instance = dataset.instances[i]
        cell_id_template = instance[1]['table_ids'][0][0]
        table_id = cell_id_template.split('_')[0] + '_table_' + cell_id_template.split('_')[-3]
        table_score_by_id[instance[1]['id']].append([table_id, score.cpu().numpy()])
    return table_score_by_id

def compute_table_retrieval(table_score_by_id, gold_evidence_by_id, args):
    recall_all = []
    for id in table_score_by_id:
        table_score_by_id[id].sort(key = lambda x: x[1], reverse = True)
        predicted_tables = list(list(zip(*table_score_by_id[id]))[0][:args.max_tabs])
        table_score_by_id[id] = predicted_tables
        if not gold_evidence_by_id[id]:
            continue
        recall = len(set(predicted_tables) & set(gold_evidence_by_id[id])) / len(set(gold_evidence_by_id[id]))
        recall_all.append(recall)
    if len(recall_all) > 0:
        recall_result = average(recall_all)
    else:
        recall_result = 'N/A'
    return recall_result, table_score_by_id

def get_gold_table_evidence(datatype, input_path):
    gold_table_by_id = {}
    if datatype in ['train', 'dev']:
        inpath = '{0}/{1}.jsonl'.format(input_path, datatype)
        annotation_processor = AnnotationProcessor(inpath)
        for anno in annotation_processor:
            id = anno.id
            evidence = anno.get_evidence(flat = True)
            gold_table_by_id[id] = list(set(evi.split('_')[0] + '_table_' + evi.split('_')[-3] for evi in evidence if '_cell_' in evi))
    elif datatype == 'test':
        inpath = '{0}/{1}.jsonl'.format(input_path, datatype)
        annotation_processor = AnnotationProcessor(inpath)
        for anno in annotation_processor:
            id = anno.id
            gold_table_by_id[id] = []
    else:
        raise NotImplementedError
    return gold_table_by_id



def main():
    use_schedular = True
    args = get_parser().parse_args()
    print(args)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tapas_path)
    
    valid_data = RC_Generator(args.input_path, tokenizer, 'dev', args)
    valid_data.get_data_for_epoch(train = False)
    gold_evidence_by_id = get_gold_table_evidence('dev', args.input_path)
    valid_data_loader = DataLoader(valid_data, batch_size = 1, shuffle = False, collate_fn = collate_fn)

    model = RC_MLP_Retriever(args).to(args.device)
    

    
    row_criterion = nn.NLLLoss()
    col_criterion = nn.NLLLoss()
    table_criterion = nn.MarginRankingLoss(margin = 1)
    col_metric = EvalMetric()
    row_metric = EvalMetric()
    
    train_data = RC_Generator(args.input_path, tokenizer, 'train', args)
    train_data.get_data_for_epoch(train = True)
    train_data_loader = DataLoader(train_data, args.batch_size, shuffle = True, collate_fn = collate_fn)

    optimizer = get_optimizer(model, lr = args.lr, weight_decay = args.weight_decay, fix_bert = args.fix_bert)
    global_step = 0
    tb = SummaryWriter( )
    if use_schedular:
        total_steps = int(args.max_epoch * len(train_data_loader)) // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer
                                                    , num_warmup_steps=int(total_steps * args.warm_rate)
                                                    , num_training_steps=total_steps)
    table_recall_flag = 0
    
    for epoch in range(args.max_epoch):
        train_data.get_data_for_epoch(train = True)
        model.train()
        row_pred_epoch = []
        col_pred_epoch = []
        row_gold_epoch = []
        col_gold_epoch = []
        
        loss_sum = 0
        loss_epoch = []

        for ii, batch in tqdm(enumerate(train_data_loader)):
            batch = [item.to(args.device) for item in batch]
            input_data = batch[:-2]

            try:
                res = model(input_data)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                res = model(input_data)

            col_res, row_res, pos_table_scores, neg_table_scores = res
            col_labels, row_labels= batch[-2:]
            golds = torch.ones_like(pos_table_scores)

            col_loss = col_criterion(col_res, col_labels)
            row_loss = row_criterion(row_res, row_labels)
            table_loss = table_criterion(pos_table_scores, neg_table_scores, golds)

            loss = args.alpha * col_loss + args.beta * row_loss + args.gamma * table_loss

            acc, recall, precision = cal_matrix(col_res, col_labels, col_pred_epoch, col_gold_epoch, args.col_threshold, args)
            col_metric.meter_add(acc, recall, precision, col_loss.item())

            acc, recall, precision = cal_matrix(row_res, row_labels, row_pred_epoch, row_gold_epoch, args.row_threshold, args)
            row_metric.meter_add(acc, recall, precision, row_loss.item())


            global_step += 1
            loss_epoch.append(loss.item())
            loss = loss / args.gradient_accumulation_steps
            loss_sum += loss.item()
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                if use_schedular:
                    lrs = scheduler.get_last_lr()
                    tb.add_scalars("learning_rates", {"bert_lr": lrs[0], "no_bert_lr": lrs[-1]}, global_step)
                tb.add_scalar("train_loss", loss.item(), global_step)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=5.0)
                optimizer.step()
                if use_schedular:
                    scheduler.step()

                grad_dict_first = print_grad(model)
                tb.add_scalars("model_grads_first", grad_dict_first, global_step)
                optimizer.zero_grad()
            if ii % args.print_freq == 0:
                print('Epoch:{0},step:{1}'.format(epoch, ii))
                freq = args.print_freq
                print('Train Loss:{:.6f}'.format(loss_sum/freq))
                print('Rows')
                row_metric.print_meter()
                print('Columns')
                col_metric.print_meter()
                loss_sum = 0
        table_scores_by_id = val(model, valid_data_loader, valid_data, args) 
        recall, predicted_tables_by_id = compute_table_retrieval(table_scores_by_id, gold_evidence_by_id, args)
        
        print(recall)
        ckpt_meta = {
            'table_recall':recall,
            'select_criterion': args.select_criterion
            }
        if recall > table_recall_flag:
            table_recall_flag = recall
            path = model.save(args.model_save_path, ckpt_meta, recall, only_max= (not args.save_all_ckpt))            

if __name__ == '__main__':
    main()