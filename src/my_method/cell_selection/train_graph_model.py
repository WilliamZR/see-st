import argparse
import sys

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

def collate_fn(batch):
    sent_input_ids, sent_attention_mask\
        , table_input_ids, table_attention_masks, table_token_type_ids\
        , row_pooling_matrix, col_pooling_matrix,sent_nums, row_nums, col_nums\
        , graph, sent_labels, row_labels, col_labels, cell_labels= map(list, zip(*batch))
    batch_node_matching_matrix = get_node_match_matrix(sent_nums,
                                                           [sum(item) for item in row_nums],
                                                           [sum(item) for item in col_nums])
    batch_sent_input, batch_sent_mask = map(torch.cat, [sent_input_ids, sent_attention_mask])
        
    table_input_ids = [torch.reshape(item, (-1, 512)) for item in table_input_ids if item != []]
    table_attention_masks = [torch.reshape(item, (-1, 512)) for item in table_attention_masks if item != []]
    table_token_type_ids = [torch.reshape(item, (-1, 512, 7)) for item in table_token_type_ids if item != []]
    if table_input_ids != []:
        batch_table_ids = torch.cat(table_input_ids)
        batch_table_attention_masks = torch.cat(table_attention_masks)
        batch_table_token_type_ids = torch.cat(table_token_type_ids)

        row_pooling_matrix = [item for item in row_pooling_matrix if item != []]
        col_pooling_matrix = [item for item in col_pooling_matrix if item != []]
        batch_row_pooling_matrix = connect_pooling_matrix(row_pooling_matrix)
        batch_col_pooling_matrix = connect_pooling_matrix(col_pooling_matrix)
    else:
        batch_table_ids = torch.tensor([])
        batch_table_attention_masks = torch.tensor([])
        batch_table_token_type_ids = torch.tensor([])
        batch_row_pooling_matrix = torch.tensor([])
        batch_col_pooling_matrix = torch.tensor([])

    if len(graph) == 1:
        batch_graph = graph[0]
    else:
        batch_graph = dgl.batch(graph)
    
    batch_sent_labels, batch_row_labels, batch_col_labels, batch_cell_labels\
    , batch_row_num, batch_col_num\
        = map(batch_label, [sent_labels, row_labels, col_labels, cell_labels, row_nums, col_nums])
    batch_cell_mask = cell_mask_generation(batch_row_num, batch_col_num)
    batch_cell_mask = batch_cell_mask.bool()

    return batch_sent_input, batch_sent_mask\
        , batch_table_ids, batch_table_attention_masks, batch_table_token_type_ids\
        ,batch_row_pooling_matrix, batch_col_pooling_matrix, batch_node_matching_matrix, batch_cell_mask\
        , batch_graph, batch_sent_labels, batch_row_labels, batch_col_labels, batch_cell_labels
    
def batch_label(labels):
    return torch.tensor([i for item in labels for i in item])

@torch.no_grad()
def val(model, dataloader, dev_data, gold_evidence, args):
    model.eval()
    entry = []
    sent_recall_all = []
    cell_recall_all = []
    recall_all = 0

    label2verdict = {1: 'SUPPORTS', 0: 'NOT ENOUGH INFO', 2: 'REFUTES'}
    for ii, batch in tqdm(enumerate(dataloader)):
        ### compute logits
        batch = [item.to(args.device) for item in batch]   
        input_data = batch[:-4]
        res = model(input_data)

        sent_pred_logits, row_pred_logits, col_pred_logits, cell_pred_logits = res
        sent_pred_logits, cell_pred_logits = map(detach, [sent_pred_logits, cell_pred_logits])

        ### select evidence
        sent_ids = dev_data.raw_data[ii]['sent_row_col_id']['sent']
        table_ids_2D = dev_data.raw_data[ii]['table_ids_2D']
        cell_ids = linearize_cell_id(table_ids_2D)
        evidence_id = dev_data.raw_data[ii]['id']
        #### Sentences threshold
        sent_score = list(zip(sent_ids, sent_pred_logits[:, -1]))
        predicted_sents = [item[0] for item in sent_score if np.exp(item[1]) > args.sent_threshold]
        #### Cells top 25
        #### 
        predicted_cells = []
        if cell_pred_logits.size != 1:
            cell_score = list(zip(cell_ids, cell_pred_logits[:, -1]))
            cell_score.sort(key = lambda x : float(x[1]), reverse = True)
            sorted_cells = list(list(zip(*cell_score))[0])
            for cell in sorted_cells:
                if cell not in predicted_cells and len(predicted_cells) < 25:
                    predicted_cells.append(cell)
                if len (predicted_cells) == 25:
                    break


        ### Compute F1 and Feverous Score metrics
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
        recall = score_check(predicted_sents, predicted_cells, golds)
        recall_all += recall / len(dataloader)
        
        ### Collect data
        entry.append({
            'id': evidence_id, 
            'evidence': golds,
            'predicted_evidence': predicted_sents + predicted_cells})
        
    return recall_all, average(sent_recall_all), average(cell_recall_all), entry

if __name__ == '__main__':
    use_schedular = True
    args = get_parser().parse_args()
    print(args)
    set_seed(args.seed)

    roberta_tokenizer = AutoTokenizer.from_pretrained(args.roberta_path)
    tapas_tokenizer = AutoTokenizer.from_pretrained(args.tapas_path)


    train_data = RowColSentFusionGraphGenerator(roberta_tokenizer, tapas_tokenizer, 'train', args)
    dev_data = RowColSentFusionGraphGenerator(roberta_tokenizer, tapas_tokenizer, 'dev', args)


    train_data_loader = DataLoader(train_data, args.batch_size, shuffle= True, collate_fn= collate_fn, num_workers=args.num_workers)
    dev_data_loader = DataLoader(dev_data, batch_size = 1, shuffle = False, collate_fn = collate_fn)## batchsize = 1 for now for convinence
    annotations_dev = AnnotationProcessor('data/dev.jsonl')
    gold_evidence_by_id_dev = {anno.get_id(): anno.get_evidence() for anno in annotations_dev}
        
    model = FusionGraphModel(args)
    if args.model_load_path:
        model.load(args.model_load_path, strict = False)
    model.to(args.device)
    optimizer = get_optimizer(model, lr = args.lr, weight_decay= args.weight_decay, fix_bert = args.fix_bert)

    global_step = 0
    tb = SummaryWriter( )
    if use_schedular:
        total_steps = int(args.max_epoch * len(train_data_loader)) // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer
                                                    , num_warmup_steps=int(total_steps * args.warm_rate)
                                                    , num_training_steps=total_steps)

    metric_flag = 0
    loss_sum = 0
    sent_criterion = nn.NLLLoss()
    row_criterion = nn.NLLLoss()
    col_criterion = nn.NLLLoss()
    cell_criterion = nn.NLLLoss()

    for epoch in range(args.max_epoch):
        model.train()
        sent_pred_epoch, row_pred_epoch, col_pred_epoch, cell_pred_epoch = [], [], [], []
        sent_gold_epoch, row_gold_epoch, col_gold_epoch, cell_gold_epoch= [], [], [], []
        sent_metric, row_metric, col_metric, cell_metric = EvalMetric(), EvalMetric(), EvalMetric(), EvalMetric()
        loss_epoch = [] 

        for ii, batch in tqdm(enumerate(train_data_loader)):
            batch = [item.to(args.device) for item in batch]
            input_data, labels = batch[:-4], batch[-4:]

            try:
                res = model(input_data)
            except RuntimeError as exception:
                torch.cuda.empty_cache()
                res = model(input_data)


            sent_pred_logits, row_pred_logits, col_pred_logits, cell_pred_logits = res
            sent_labels, row_labels, col_labels, cell_labels = labels

            sent_loss = sent_criterion(sent_pred_logits, sent_labels)
            if row_pred_logits.size() != torch.Size([]):
                row_loss = row_criterion(row_pred_logits, row_labels)
                col_loss = col_criterion(col_pred_logits, col_labels)
                cell_loss = cell_criterion(cell_pred_logits, cell_labels)
            else:
                row_loss, col_loss, cell_loss = 0, 0, 0



            ### compute metric and loss for evidence extractioin
            loss_and_metric(sent_metric, sent_loss, sent_pred_logits, sent_labels, sent_pred_epoch, sent_gold_epoch, args.sent_threshold,args)

            ### Compute metric for table format evidence extraction
            if row_labels.size() != torch.Size([0]):

                loss_and_metric(row_metric, row_loss, row_pred_logits, row_labels, row_pred_epoch, row_gold_epoch, args.row_threshold,args)
                loss_and_metric(col_metric, col_loss, col_pred_logits, col_labels, col_pred_epoch, col_gold_epoch, args.col_threshold, args)
                loss_and_metric(cell_metric, cell_loss, cell_pred_logits, cell_labels, cell_pred_epoch, cell_gold_epoch, args.cell_threshold, args)

            global_step += 1
            if row_labels.size() != torch.Size([0]):
                loss = args.sent_k * sent_loss + args.row_k * row_loss + args.col_k * col_loss + args.cell_k * cell_loss
            else:
                loss = args.sent_k * sent_loss      

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
                '''
                grad_dict_second = print_grad(model, "second")
                tb.add_scalars("model_grads_second", grad_dict_second, global_step)
                '''
                optimizer.zero_grad()

            if ii % args.print_freq == 0:
                print('Epoch:{0},step:{1}'.format(epoch, ii))
                freq = args.print_freq
                print('Train Loss:{:.6f}'.format(loss_sum/freq))
                print('Sentences:', end = ':')
                sent_metric.print_meter()
                print('Rows', end  = ':')
                row_metric.print_meter()
                print('Columns', end = ':')
                col_metric.print_meter()
                print('Cells', end = ':')
                cell_metric.print_meter()
                loss_sum = 0

        print('======train step of epoch {} ======'.format(epoch))
        recall, sent_recall, cell_recall, entry = val(model, dev_data_loader, dev_data, gold_evidence_by_id_dev, args)
        print('======valid step of epoch {} ======'.format(epoch))
        print('Sentence Recall: {:.5f}'.format(sent_recall))
        print('Cell Recall: {:.5f}'.format(cell_recall))
        print('Evidence Set Recall: {:.5f}'.format(recall))

        ckpt_meta = {
            'recall': recall,
            'sentence recall': sent_recall,
            'cell_recall': cell_recall,
            'evidence_set_recall': recall
        }
        path = model.save(args.ckpt_root_dir, ckpt_meta, recall, only_max = (not args.save_all_ckpt))
        save_entry(entry, 'checkpoints', args.entry_save_file)


def print_res(preds, golds, data_type):
    print(data_type)
    acc = sum([1 if p == g else 0 for p, g in zip(preds, golds)]) / len(preds)
    recall = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(golds)
    prec = len([1 for p, g in zip(preds, golds) if p == g and p == 1]) / sum(preds) if sum(preds) else 1

    print("{}_epoch{}_accuracy:".format(data_type, epoch), acc)
    print("{}_epoch{}_precision:".format(data_type, epoch), prec)
    print("{}_epoch{}_recall:".format(data_type, epoch), recall)
    scores = compute_metrics(preds, golds)
