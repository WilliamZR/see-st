# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/10 19:32
# Description:

# !/usr/bin/python3.7
# -*-coding:utf-8-*-
# Author: Hu Nan
# CreatDate: 2022/1/4 13:47
# Description:

from transformers import TapasConfig, TapasModel, AutoModel, AutoConfig
from base_templates import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, nhid, args):
        super(SelfAttentionLayer, self).__init__()
        self.args = args
        self.args.weights = []
        self.project = nn.Sequential(
            nn.Linear(nhid, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, inputs, return_attention=False):
        attention = self.project(inputs)
        weights = F.softmax(attention.squeeze(-1), dim=1)
        # self.args.weights.append(weights.detach().cpu().numpy()[0])
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        if return_attention:
            return outputs, weights
        else:
            return outputs

class WeightedBiModalCls(BasicModule):
    def __init__(self, args):
        super(WeightedBiModalCls, self).__init__()
        self.config = TapasConfig.from_pretrained(args.bert_name, num_labels=len(args.id2label))
        hidden_size = self.config.hidden_size
        self.linear1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, len(args.id2label))

        self.config_2 = AutoConfig.from_pretrained(args.bert_name_2, num_labels=len(args.id2label))
        hidden_size_2 = self.config_2.hidden_size
        self.linear1_2 = nn.Linear(hidden_size_2, 128)
        self.relu_2 = nn.ReLU()
        self.linear2_2 = nn.Linear(128, len(args.id2label))

        # self.attention_pooling = SelfAttentionLayer(len(args.id2label))
        # self.ensemble_linear = nn.Linear(2*len(args.id2label), len(args.id2label))

        self.init_weights()
        self.args = args

        self.bert = TapasModel.from_pretrained(args.bert_name, config=self.config)
        self.dropout = nn.Dropout(args.dropout)

        self.bert2 = AutoModel.from_pretrained(args.bert_name_2, config=self.config_2)
        self.dropout_2 = nn.Dropout(args.dropout)

        self.count_parameters()


    def forward(self, batch, args, test_mode):
        raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2, merge_weight, labels = batch

        # table
        outputs = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = outputs[1]#.view(input_shape[0], -1)
        hg = self.dropout(output)
        hg = self.relu(self.linear1(hg))
        hg = self.linear2(hg)
        hg1 = hg

        # text
        outputs = self.bert2(input_ids2, attention_mask=attention_mask2)
        output = outputs[1]  # .view(input_shape[0], -1)
        hg = self.dropout_2(output)
        hg = self.relu_2(self.linear1_2(hg))
        hg = self.linear2_2(hg)
        hg2 = hg

        # hg = hg1 * merge_weight[:,0:1] + hg2 * merge_weight[:,1:2]

        pred_logits = torch.log_softmax(hg, dim=-1)
        golds = labels

        return pred_logits, golds
