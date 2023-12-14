### graph_model.py
### Wu Zirui
### 20230114
### A mixed graph of sentences, rows and columns
### 

from transformers import AutoModel
from base_templates import BasicModule
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch import AvgPooling, SortPooling
class FusionGraphModel(BasicModule):
    def __init__(self, args):
        super(FusionGraphModel, self).__init__()
        self.bert = AutoModel.from_pretrained(args.roberta_path)
        self.table_bert = AutoModel.from_pretrained(args.tapas_path)
        self.relu = nn.ReLU()

        self.row_linear = nn.Linear(768, 768)
        self.row_linear1 = nn.Linear(768, 128)
        self.row_linear2 = nn.Linear(128, 2)

        self.col_linear = nn.Linear(768, 768)
        self.col_linear1 = nn.Linear(768, 128)
        self.col_linear2 = nn.Linear(128, 2)

        self.sent_linear = nn.Linear(768, 768)
        self.sent_linear1 = nn.Linear(768, 128)
        self.sent_linear2 = nn.Linear(128, 2)

        self.graph_kernel = GATConv(768, 768, 1, residual = True)
        self.args = args



    def forward(self, batch):
        sent_input_ids, sent_input_mask\
            , table_input_ids, table_input_mask, table_token_type_ids\
            , row_pooling_matrix, col_pooling_matrix, node_matching_matrix ,cell_mask, graph = batch
        # row num is an array of row nums of each table
        sent_output = self.bert(sent_input_ids, sent_input_mask)[1]
        sent_embs = self.relu(self.sent_linear(sent_output))
        batch_num_sentences = sent_embs.size()[0]

        if table_input_ids.size() == torch.Size([0]):
            row_embs = torch.tensor([]).to(self.args.device)
            col_embs = torch.tensor([]).to(self.args.device)
        else:
            table_output = self.table_bert(table_input_ids, table_input_mask, table_token_type_ids).last_hidden_state
            table_output = torch.reshape(table_output, (-1, 768))

            row_embs = self.relu(self.row_linear(table_output))
            row_embs = torch.matmul(row_pooling_matrix, row_embs)
            batch_num_rows = row_embs.size()[0]

            col_embs = self.relu(self.col_linear(table_output))
            col_embs = torch.matmul(col_pooling_matrix, col_embs)
            batch_num_cols = col_embs.size()[0]

        graph_embs = torch.cat([sent_embs, row_embs, col_embs], dim = 0)
  #      index = torch.cat([torch.nonzero(graph.ndata['t']==a).squeeze(1) for a in range(3)])
 #       graph_embs = graph_embs[index]
#        graph_embs = torch.zeros(sent_embs.size()[0] + row_embs.size()[0] + col_embs.size()[0], 768).to(self.args.device)
#        graph_embs[torch.nonzero(graph.ndata['t']==0).squeeze(1), :] = sent_embs
#        if table_input_ids.size() != torch.Size([0]):
#            graph_embs[torch.nonzero(graph.ndata['t']==1).squeeze(1), :] = row_embs
#            graph_embs[torch.nonzero(graph.ndata['t']==2).squeeze(1), :] = col_embs
        graph_embs = self.graph_kernel(graph, graph_embs)

        ### Make Predictions for Sentence Nodes
        sent_h = graph_embs[:batch_num_sentences]
        sent_logits = self.sent_linear2(self.relu(self.sent_linear1(sent_h)))
        sent_logits = torch.softmax(sent_logits.view(-1, 2), dim = -1)

        if table_input_ids.size() == torch.Size([0]):
            row_logits = torch.tensor(1).to(self.args.device)
            col_logits = torch.tensor(1).to(self.args.device)
            cell_logits = torch.tensor(1).to(self.args.device)
 
        else:
            row_h = graph_embs[batch_num_sentences:-batch_num_cols]
            col_h = graph_embs[-batch_num_cols:]
            row_logits = self.row_linear2(self.relu(self.row_linear1(row_h)))
            col_logits = self.col_linear2(self.relu(self.col_linear1(col_h)))

            #### Cell Selection Probability
            #### some cells are redundant, neglect redundance here.
            #### multiplicate row logit and col logit to get a matrix D(row_num, col_num)
            #### Then use a mask to extract cell probabilities
            row_logits = torch.softmax(row_logits.view(-1, 2), dim = -1)
            col_logits = torch.softmax(col_logits.view(-1, 2), dim = -1)
            cell_logits = torch.matmul(row_logits[:, -1].view(-1, 1), col_logits[:, -1].view(1, -1))
            cell_logits = torch.masked_select(cell_logits, cell_mask).view(-1, 1)
            cell_logits = cell_logits / torch.sum(cell_logits)
            cell_logits = torch.cat([1 - cell_logits, cell_logits], dim = -1)

        
        return torch.log(sent_logits), torch.log(row_logits), torch.log(col_logits), torch.log(cell_logits)

