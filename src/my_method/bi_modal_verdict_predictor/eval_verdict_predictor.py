# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/3 19:34
# Description:
# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2020/12/16 18:55
# Description:
import json
import os
import sys
os.chdir('/home/wuzr/feverous')
sys.path.append('src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizer

import bi_modal_config as config
from base_templates import BasePreprocessor
from bi_modal_arg_parser import get_parser

from bi_modal_cls import BiModalCls
from bi_modal_cls_concat import BiModalClsConcat
from weighted_bi_modal_cls import WeightedBiModalCls
from bi_modal_generator import BiModalGenerator
from weighted_bi_modal_generator import WeightedBiModalGenerator
from bi_modal_coattention_generator import BiModalCoAttentionGenerator, BiModalHierarchialCoAttentionGenerator  
from bi_modal_coattention import BiModalCoAttention, BiModalHierarchicalCoAttention, BiModalCoAttentionEnsemble

from my_utils import set_seed, save_jsonl_data, compute_metrics, load_jsonl_data



def main(args, save = True):
    if args == None:
        args = get_parser().parse_args()
    if not args.test_ckpt:
        # args.test_ckpt = "BiModalCls_0104_20:48:12" # gold
        # # args.test_ckpt = "BiModalCls_0105_00:07:34" # both
        #args.test_ckpt = "BiModalCls_0508_21:47:48" 
        args.test_ckpt = 'BiModalCls_0523_21:23:48'

    assert args.test_ckpt is not None

    test_path = os.path.join(args.ckpt_root_dir, args.test_ckpt)
    args.test_path = test_path
    meta_path = os.path.join(test_path, "ckpt.meta")
    print(meta_path)
    
    ckpt_model_type = args.test_ckpt.split('_')[0]
    if not args.model_type:
        args.model_type = ckpt_model_type
    assert args.model_type == ckpt_model_type, print(args.model_type, ckpt_model_type)
        
    set_seed(args.seed)

    # 1 table 2 text
    args.bert_name_2 = '/home/hunan/bert_weights/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    tokenizer2 = RobertaTokenizer.from_pretrained(args.bert_name_2, model_max_length=512)

    preprocessor = BasePreprocessor(args)

    args.label2id = config.label2idx
    args.id2label = dict(zip([value for _, value in args.label2id.items()], [key for key, _ in args.label2id.items()]))
    args.config = config
    args.tokenizer = [tokenizer, tokenizer2]


    if args.model_type == "BiModalCls":
        model = BiModalCls(args)
        data_generator = BiModalGenerator
    elif args.model_type == "BiModalClsConcat":
        model = BiModalClsConcat(args)
        data_generator = BiModalCoAttentionGenerator
    elif args.model_type == "WeightedBiModalCls":
        model = WeightedBiModalCls(args)
        data_generator = WeightedBiModalGenerator
    elif args.model_type == "BiModalCoAttention":
        model = BiModalCoAttention(args)
        data_generator = BiModalCoAttentionGenerator
    elif args.model_type == 'BiModalHierarchicalCoAttention':
        model = BiModalHierarchicalCoAttention(args)
        data_generator = BiModalHierarchialCoAttentionGenerator
    elif args.model_type == 'BiModalCoAttentionEnsemble':
        model = BiModalCoAttentionEnsemble(args)
        data_generator = BiModalCoAttentionGenerator
    else:
        assert False, args.model_type

    train_data, valid_data, test_data = preprocessor.process(args.data_dir, args.cache_dir
                                                    , data_generator, args.tokenizer, dataset=["dev_retrieved"])

    collate_fn = data_generator.collate_fn()

    ckpt_meta = model.load(test_path)

    model.to(args.device)
    criterion = nn.NLLLoss()

    if valid_data:
        args.data_type = "dev"
        # valid_data.shuffle_evis()
        #valid_data.reverse_evis()
        val_dataloader = DataLoader(valid_data, args.batch_size, collate_fn=collate_fn, shuffle=False)
        val_loss, val_acc, val_preds, val_golds = _val(model, val_dataloader, criterion, None, None, args)
        print(val_acc)
        assert len(val_preds) == len(val_golds)
        odata = []
        for pl, entry in zip(val_preds, valid_data.raw_data):
            oentry = {}
            oentry["evidence"] = entry["evidence"]
            oentry["predicted_evidence"] = entry["predicted_evidence"]
            oentry["predicted_label"] = args.id2label[pl]
            oentry["label"] = entry["label"]
            odata.append(oentry)
        save_jsonl_data(odata, "data/dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl")
        os.system("python evaluation/evaluate.py --input_path data/dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl")

        input_path = "data/dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl"
        data = load_jsonl_data(input_path)
        assert len(val_preds) == len(data), print(len(val_preds), len(data))
        for entry, pred in zip(data, val_preds):
            entry["predicted_label"] = config.idx2label[pred]
        save_jsonl_data(data, f"./submission_bimodal_{args.test_ckpt}.dev.jsonl")

    if train_data:
        args.data_type = "train"
        train_dataloader = DataLoader(train_data, args.batch_size, collate_fn=collate_fn, shuffle=False)
        train_loss, train_acc, train_preds, train_golds = _val(model, train_dataloader, criterion, None, None, args)
        print(train_acc)

    if test_data:
        args.data_type = "test"
        #test_data.shuffle_evis()
        test_dataloader = DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=False)
        _, _, test_preds, _ = _val(model, test_dataloader, criterion, None, None, args)
        
        odata = []
        for pl, entry in zip(test_preds, test_data.raw_data):
            oentry = {}
            oentry["predicted_evidence"] = entry["predicted_evidence"]
            oentry["predicted_label"] = args.id2label[pl]
            odata.append(oentry)
        save_jsonl_data(odata, "data/test.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl")
        #input_path = "data/test_temp.jsonl"
        #data = load_jsonl_data(input_path)
        #assert len(test_preds) == len(data)
        #for entry, pred in zip(data, test_preds):
        #    entry["predicted_label"] = config.idx2label[pred]
        #save_jsonl_data(data, f"submission_bimodal_{args.test_ckpt}.jsonl")

ENCODING = 'utf-8'
def test_collector(truth_file, test_preds, result_file, threshold, args):
    fin = open(truth_file, 'rb')

    truth_list = []
    for line in fin:
        arr = line.decode(ENCODING).strip('\r\n').split('\t')
        label = arr[0]
        evidence = arr[1]
        claim = arr[2]
        claim_num = arr[3]
        article = arr[4]
        article_index = arr[5]
        confidence = float(arr[6])

        if confidence >= threshold:
            truth_list.append([claim_num, label, evidence, claim, article, article_index])
    fin.close()

    claim2info = {}
    for item in truth_list:
        claim_num = int(item[0])
        if claim_num not in claim2info:
            claim2info[claim_num] = []
        claim2info[claim_num].append(item[1:])

    claim2id = {}
    fin = open(result_file, 'rb')
    lines = fin.readlines()
    for i in range(len(lines)):
        line = lines[i]
        claim2id[i] = json.loads(line)['id']
    fin.close()

    answers = []
    cnt = -1
    for i in range(0, 19998):
        answer = {}
        answer['id'] = claim2id[i]
        if i not in claim2info:
            answer = {"predicted_label": "NOT ENOUGH INFO",  "predicted_evidence": []}
            answers.append(answer)
            continue
        cnt += 1
        answer["predicted_label"] = get_predicted_label(test_preds[cnt])
        answer["predicted_evidence"] = []
        for item in claim2info[i]:
            answer["predicted_evidence"].append([item[3], int(item[4])])

        answers.append(answer)

    fout = open(os.path.join(args.test_path, 'predictions.jsonl'), 'wb')
    for answer in answers:
        fout.write(('%s\r\n' % json.dumps(answer)).encode(ENCODING))
    fout.close()

def postpro_preds(args = None):
    if args == None:
        args = get_parser().parse_args()
        if args.test_ckpt == None:
            assert False
        test_path = os.path.join(args.ckpt_root_dir, args.test_ckpt)
        args.test_path = test_path

    from my_utils import save_jsonl_data, load_jsonl_data
    temp_data = load_jsonl_data(os.path.join(args.data_dir, "test_temp.jsonl"))

    meta_path = os.path.join(args.test_path, "ckpt.meta")
    ckpt_meta = json.load(open(meta_path, "r", encoding="utf-8"))
    test_preds = ckpt_meta["test_preds"]

    assert len(temp_data) == len(test_preds)

    preds_data_dict = {}
    for i in range(len(temp_data)):
        temp_data[i]["predicted_label"] = get_predicted_label(test_preds[i])
        preds_data_dict[temp_data[i]["id"]] = temp_data[i]
    predictions = []
    order_data = load_jsonl_data(os.path.join(args.data_dir, "test_order.json"))
    for od in order_data:
        predictions.append(preds_data_dict[od["id"]])
    save_jsonl_data(predictions, os.path.join(args.test_path, 'predictions.jsonl'))

def get_predicted_label(label_idx):
    labels = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    return labels[label_idx]

@torch.no_grad()
def _val(model, dataloader, criterion, tokenizer, preprocessor, args):
    """
    计算模型在验证集上的准确率等信息
    """
    loss_sum = 0
    acc_sum = 0
    preds_epoch = []
    golds_epoch = []

    model.eval()
    for ii, data_entry in tqdm(enumerate(dataloader)):
        res = model(data_entry, args, test_mode = True)
        if len(res) == 2:
            logits, golds = res
            loss = criterion(logits, golds)
        elif len(res) == 3:
            logits, golds, apd_loss = res
            loss = criterion(logits, golds) + apd_loss
        else:
            assert False
        preds = logits.topk(k=1, dim=-1)[1].squeeze(-1)
        acc = sum(preds==golds).item()/len(preds)

        loss_sum += loss.item() * len(preds)
        acc_sum += acc*len(preds)

        preds_epoch.extend(list(preds.cpu().numpy().tolist()))
        golds_epoch.extend(list(golds.cpu().numpy().tolist()))

#    scores = compute_metrics(preds_epoch, golds_epoch, target_names = ['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES'])

    los = loss_sum / dataloader.dataset.__len__()
    acc = acc_sum / dataloader.dataset.__len__()

    return los, acc, preds_epoch, golds_epoch


def print_meta(ckpt_meta):
    print("train_acc:", ckpt_meta["train_acc"])
    print("val_acc:", ckpt_meta["val_acc"])
    #print("test_acc", ckpt_meta["test_acc"])

if __name__ == "__main__":
    main(None, save = True)
    #postpro_preds()
