### bi_modal_coattention.py
### Wu Zirui
### 2023/6/1
from transformers import TapasConfig, TapasModel, AutoModel, AutoConfig
from base_templates import BasicModule
import torch
import torch.nn as nn

class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CoAttention, self).__init__()
        self.hidden_size = hidden_size
        self.affinity_matrix = nn.Linear(hidden_size, hidden_size)
        self.sent_linear = nn.Linear(hidden_size, hidden_size)
        self.table_linear = nn.Linear(hidden_size, hidden_size)

        self.sent_linear2 = nn.Linear(hidden_size, 1)
        self.table_linear2 = nn.Linear(hidden_size, 1)

        self.sent_tanh = nn.Tanh()
        self.tab_tanh = nn.Tanh()

class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CoAttention, self).__init__()
        self.hidden_size = hidden_size
        self.affinity_matrix = nn.Linear(hidden_size, hidden_size, bias = False)
        self.sent_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        self.table_linear = nn.Linear(hidden_size, hidden_size, bias = False)

        self.sent_linear2 = nn.Linear(hidden_size, 1, bias = False)
        self.table_linear2 = nn.Linear(hidden_size, 1, bias = False)

        self.sent_tanh = nn.Tanh()
        self.tab_tanh = nn.Tanh()

    def forward(self, sent_embedding, table_embedding, sent_mask, table_mask):
        ### input: sentence embedding and table embedding
        ### size [#batch_size, #sentence_length/#table_length, #hidden_size]

        ### Affinity matrix is computed as the dot product of the sentence and parameter matrix and table embeddings
        ### F = tanh(S(t)WT)
        ### S(t) is transpose the sentence embedding, W is the parameter matrix, T is the table embedding
        ### F is the affinity matrix
        ### affinity_matrix size [#batch_size, #sentence_length, #table_length]
        sent_mask, table_mask = sent_mask.float(), table_mask.float()
        mask = torch.matmul(sent_mask.unsqueeze(-1), table_mask.unsqueeze(-2))
        affinity_matrix = torch.matmul(sent_embedding, self.affinity_matrix(table_embedding).transpose(-1, -2))
        affinity_matrix = affinity_matrix * mask

        ### H_{sent} = tanh(W_{s} S + (W_{t} T) F^{T})
        ### H_{table} = tanh(W_{t} T + (W_{s} S) F)
        h_sent = self.sent_tanh(self.sent_linear(sent_embedding) + torch.matmul(affinity_matrix, self.table_linear(table_embedding)))
        h_table = self.tab_tanh(self.table_linear(table_embedding) + torch.matmul(affinity_matrix.transpose(-1,-2), self.sent_linear(sent_embedding)))
        h_sent = self.sent_linear2(h_sent)
        h_table = self.table_linear2(h_table)
        
        h_sent = torch.masked_fill(h_sent, sent_mask.unsqueeze(-1) == 0, float('-inf'))
        h_table = torch.masked_fill(h_table, table_mask.unsqueeze(-1) == 0, float('-inf'))
        
        ### Size: [#batch_size, #sentence_length/#table_length, 1]
        p_sent = torch.softmax(h_sent, dim=-2)
        p_table = torch.softmax(h_table, dim=-2)
        return p_sent, p_table
    

class BiModalCoAttention(BasicModule):
    def __init__(self, args):
        super(BiModalCoAttention, self).__init__()
        self.config = TapasConfig.from_pretrained(args.bert_name, num_labels=len(args.id2label))
        hidden_size = self.config.hidden_size


        self.config_2 = AutoConfig.from_pretrained(args.bert_name_2, num_labels=len(args.id2label))
        hidden_size_2 = self.config_2.hidden_size
        assert hidden_size == hidden_size_2, (hidden_size, hidden_size_2)
        self.hidden_size = hidden_size
        self.word_linear = nn.Linear(hidden_size * 2, hidden_size * 2)

        self.linear1 = nn.Linear(hidden_size * 4, 128)
#        self.linear1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, len(args.id2label))

        self.init_weights()
        self.args = args

        self.bert = TapasModel.from_pretrained(args.bert_name, config=self.config)
        self.dropout = nn.Dropout(args.dropout)

        self.bert2 = AutoModel.from_pretrained(args.bert_name_2, config=self.config_2)
        self.dropout_2 = nn.Dropout(args.dropout)

        self.word_level_coattention = CoAttention(hidden_size)

        self.count_parameters()


    def forward(self, batch, args, test_mode):
        raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2, labels = batch

        ### hg1:embeddings for table-format embeddings
        outputs = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        table_embedding = outputs[1]
        output = outputs.last_hidden_state#.view(input_shape[0], -1)

        hg_table = output

        ### hg2:embeddings for text-format embeddings
        ### embedding_vectors = outputs.last_hidden_state[0][torch.where(torch.tensor(input_ids) == tokenizer.sep_token_id)]

        outputs = self.bert2(input_ids2, attention_mask=attention_mask2)
        sent_embedding = outputs[1]
        output = outputs.last_hidden_state  # .view(input_shape[0], -1)

        hg_sent = output

        p_sent, p_table = self.word_level_coattention(hg_sent, hg_table, attention_mask2, attention_mask)
        hg_table = torch.matmul(p_table.transpose(-1, -2), hg_table)
        hg_sent = torch.matmul(p_sent.transpose(-1, -2), hg_sent)

        hg = self.word_linear(torch.cat([hg_table, hg_sent], dim=-1))
        hg_cont = torch.cat([table_embedding, sent_embedding],dim = -1).unsqueeze(1)

        hg = self.dropout(hg)
        hg_cont = self.dropout_2(hg_cont)
        
        hg = self.linear1(torch.cat([hg, hg_cont], dim = -1))
        #hg = self.linear1(hg + hg_cont)

        hg = self.linear2(self.relu(hg))

        pred_logits = torch.log_softmax(hg, dim=-1)
        golds = labels

        return pred_logits.view(-1, 3), golds
    

class BiModalHierarchicalCoAttention(BasicModule):
    def __init__(self, args):
        super(BiModalHierarchicalCoAttention, self).__init__()
        self.config = TapasConfig.from_pretrained(args.bert_name, num_labels=len(args.id2label))
        hidden_size = self.config.hidden_size


        self.config_2 = AutoConfig.from_pretrained(args.bert_name_2, num_labels=len(args.id2label))
        hidden_size_2 = self.config_2.hidden_size
        assert hidden_size == hidden_size_2, (hidden_size, hidden_size_2)

        self.token_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.sub_linear = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.linear1 = nn.Linear(hidden_size * 6, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, len(args.id2label))

        self.init_weights()
        self.args = args

        self.bert = TapasModel.from_pretrained(args.bert_name, config=self.config)
        self.dropout = nn.Dropout(args.dropout)

        self.bert2 = AutoModel.from_pretrained(args.bert_name_2, config=self.config_2)
        self.dropout_2 = nn.Dropout(args.dropout)

        self.word_level_coattention = CoAttention(hidden_size)
        self.sent_level_coattention = CoAttention(hidden_size)

        self.count_parameters()

    def forward(self, batch, args, test_mode):
        raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2, table_index, table_mask, sent_index, sent_mask, labels = batch
        sent_index, sent_mask, table_index, table_mask = sent_index.to(self.args.device), sent_mask.to(self.args.device), table_index.to(self.args.device), table_mask.to(self.args.device)

        outputs = self.bert2(input_ids2, attention_mask=attention_mask2)
        sent_embedding = outputs[1]
        output = outputs.last_hidden_state  # .view(input_shape[0], -1)
        hg_sent_sub = torch.gather(output, 1, sent_index.unsqueeze(-1).expand(-1, -1, output.shape[-1]))

        hg_sent_token = output

        outputs = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        table_embedding = outputs[1]
        output = outputs.last_hidden_state#.view(input_shape[0], -1)
        hg_table_sub = torch.gather(output, 1, table_index.unsqueeze(-1).expand(-1, -1, output.shape[-1]))
        hg_table_token = output

        ### hg2:embeddings for text-format embeddings
        ### embedding_vectors = outputs.last_hidden_state[0][torch.where(torch.tensor(input_ids) == tokenizer.sep_token_id)]



        p_sent, p_table = self.word_level_coattention(hg_sent_token, hg_table_token, attention_mask2, attention_mask)
        hg_table_token   = torch.matmul(p_table.transpose(-1, -2), hg_table_token)
        hg_sent_token = torch.matmul(p_sent.transpose(-1, -2), hg_sent_token)

        p_sent, p_table = self.sent_level_coattention(hg_sent_sub, hg_table_sub, sent_mask, table_mask)  
        ### 检验p_sent中是否有nan
        if torch.isnan(p_sent).any():
            hg_sent_sub = sent_embedding.unsqueeze(1)
            print('Replace nan value')
        else:
            hg_sent_sub = torch.matmul(p_sent.transpose(-1, -2), hg_sent_sub)
            
        if torch.isnan(p_table).any():
            hg_table_sub = table_embedding.unsqueeze(1)
        else:
            hg_table_sub = torch.matmul(p_table.transpose(-1, -2), hg_table_sub)
        
    

        hg_token = self.token_linear(torch.cat([hg_table_token, hg_sent_token], dim=-1))
        hg_sub = torch.cat([hg_table_sub, hg_sent_sub], dim=-1)
        hg_sub = self.sub_linear(torch.cat([hg_token, hg_sub], dim=-1))
        hg_cont = torch.cat([table_embedding, sent_embedding],dim = -1).unsqueeze(1) 
        hg = torch.cat([hg_cont, hg_sub], dim = -1)
        hg = self.linear1(hg)
        hg = self.linear2(self.relu(hg))

        pred_logits = torch.log_softmax(hg, dim=-1)
        golds = labels

        return pred_logits.view(-1, 3), golds


class BiModalCoAttentionEnsemble(BasicModule):
    def __init__(self, args):
        super().__init__()

        self.config = TapasConfig.from_pretrained(args.bert_name, num_labels=len(args.id2label))
        hidden_size = self.config.hidden_size

        self.config_2 = AutoConfig.from_pretrained(args.bert_name_2, num_labels=len(args.id2label))
        hidden_size_2 = self.config_2.hidden_size
        assert hidden_size == hidden_size_2, (hidden_size, hidden_size_2)

        self.init_weights()
        self.args = args

        self.bert = TapasModel.from_pretrained(args.bert_name, config=self.config)
        self.dropout = nn.Dropout(args.dropout)

        self.bert2 = AutoModel.from_pretrained(args.bert_name_2, config=self.config_2)
        self.dropout_2 = nn.Dropout(args.dropout)

        self.word_level_coattention = CoAttention(hidden_size)
        
        self.token_linear = nn.Linear(hidden_size * 2, 128)
        self.linear = nn.Linear(hidden_size * 2, 128)
        self.token_linear2 = nn.Linear(128, len(self.args.id2label))
        self.linear2 = nn.Linear(128, len(self.args.id2label))
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.ensemble_linear = nn.Linear(2 * len(args.id2label), len(args.id2label)) 
        self.count_parameters()
    
    def forward(self, batch, args, test_mode):
        raw_data, input_ids, attention_mask, token_type_ids,  input_ids2, attention_mask2, labels = batch

        ### hg1:embeddings for table-format embeddings
        outputs = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        table_embedding = outputs[1]
        output = outputs.last_hidden_state#.view(input_shape[0], -1)

        hg_table = output

        ### hg2:embeddings for text-format embeddings
        ### embedding_vectors = outputs.last_hidden_state[0][torch.where(torch.tensor(input_ids) == tokenizer.sep_token_id)]

        outputs = self.bert2(input_ids2, attention_mask=attention_mask2)
        sent_embedding = outputs[1]
        output = outputs.last_hidden_state  # .view(input_shape[0], -1)

        hg_sent = output

        p_sent, p_table = self.word_level_coattention(hg_sent, hg_table, attention_mask2, attention_mask)
        hg_table = torch.matmul(p_table.transpose(-1, -2), hg_table)
        hg_sent = torch.matmul(p_sent.transpose(-1, -2), hg_sent)

        hg_token = torch.cat([hg_table, hg_sent], dim=-1)
        hg_cont = torch.cat([table_embedding, sent_embedding],dim = -1).unsqueeze(1)

        hg_token = self.dropout(hg_token)
        hg_cont = self.dropout_2(hg_cont)

        hg_token = self.token_linear2(self.relu(self.token_linear(hg_token)))
        hg_cont = self.linear2(self.relu2(self.linear(hg_cont)))

        hg = self.ensemble_linear(torch.cat([hg_token, hg_cont], dim = -1))

        pred_logits = torch.log_softmax(hg, dim=-1)
        golds = labels

        return pred_logits.view(-1, 3), golds