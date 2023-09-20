### rc_model.py
### Wu Zirui
### 20221027
### Use TAPAS for encoding
### average pooling + MLP for prediction for now

from transformers import AutoModel
from base_templates import BasicModule
import torch
import torch.nn as nn


class RC_MLP_Retriever(BasicModule):
    def __init__(self, args):
        super(RC_MLP_Retriever, self).__init__()
        self.bert = AutoModel.from_pretrained(args.tapas_path)
        self.relu = nn.ReLU()
        self.table_linear = nn.Linear(768, 768)
        self.table_relu = nn.ReLU()
        self.table_linear2 = nn.Linear(768, 2)

        self.row_linear = nn.Linear(768, 768)
        self.row_relu = nn.ReLU()
        self.row_linear2 = nn.Linear(768, 2)

        self.col_linear = nn.Linear(768, 768)
        self.col_relu = nn.ReLU()
        self.col_linear2 = nn.Linear(768, 2)

    def forward(self, batch, test_mode = False):
        table_input_ids, table_input_mask, table_token_type_ids, row_pooling_matrix, col_pooling_matrix = batch
        table_output = self.bert(table_input_ids, attention_mask = table_input_mask, token_type_ids = table_token_type_ids)

        table_embedding = table_output[1]
        table_output = table_output.last_hidden_state
        table_output = torch.reshape(table_output, (-1, 768))

        row_embeddings = torch.matmul(row_pooling_matrix, table_output)
        col_embeddings = torch.matmul(col_pooling_matrix, table_output)
        
        row_h = self.row_relu(self.row_linear(row_embeddings))
        row_logits = torch.log_softmax(self.row_linear2(row_h), dim = 1)
        col_h = self.col_relu(self.col_linear(col_embeddings))
        col_logits = torch.log_softmax(self.col_linear2(col_h), dim = 1)

        table_h = self.table_relu(self.table_linear(table_embedding))
        table_h = self.table_linear2(table_h)
        if test_mode:
            pos_table_scores = torch.sigmoid(table_h).view(-1)
            neg_table_scores = None
        else:
            scores = torch.sigmoid(table_h).view(-1, 2)
            pos_table_scores = scores[0]
            neg_table_scores = scores[1]
        return row_logits, col_logits, pos_table_scores, neg_table_scores

