### rc_retriever_parser.py
### Wu Zirui
###

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,default='data',help='/path/to/data')
    parser.add_argument('--tapas_path', type = str, default= '/home/wuzr/bert_weights/tapas-base')
    parser.add_argument('--model_save_path', type = str)
    parser.add_argument('--output_path', type = str)
    parser.add_argument('--device',type=str,default='cuda:1')
    parser.add_argument('--model_load_path', type = str)
    parser.add_argument('--max_tabs', type = int, default=3)
    parser.add_argument('--max_segs', type = int, default = 100)

    parser.add_argument('--wiki_path',default='/home/wuzr/feverous/data/feverous_wikiv1.db',type=str)

    parser.add_argument('--max_epoch',type=int, default = 3)

    parser.add_argument('--warm_rate', type=int,default = 0)
    parser.add_argument("--lr", default=1e-7, type=float)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--print_freq",type = int,default=100)

    parser.add_argument('--seed', default = 1234, type = int)
    parser.add_argument('--fix_bert', action = 'store_true')

    parser.add_argument('--alpha', type = float, default = 1, help = 'column loss')
    parser.add_argument('--beta', type = float, default = 1, help = 'row loss')
    parser.add_argument('--gamma', type = float, default = 1, help = 'table loss')
    parser.add_argument('--col_threshold', type = float, default = 0.25)
    parser.add_argument('--row_threshold', type = float, default = 0.25)
    parser.add_argument('--test_mode', action = 'store_true')
    parser.add_argument('--select_criterion', type = str, default= 'row*col')
    parser.add_argument('--save_all_ckpt', action = 'store_true')
    parser.add_argument('--eval_split', type = str)
    return parser

if __name__ == '__main__':
    pass
