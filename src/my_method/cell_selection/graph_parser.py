import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action = 'store_true')
    ### Path Arguments
    parser.add_argument('--roberta_path', type = str)
    parser.add_argument('--tapas_path', type = str)
    parser.add_argument('--wiki_path', type=str)
    parser.add_argument('--ckpt_root_dir', type = str, default='checkpoint')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--model_load_path', type =str)
    parser.add_argument('--entry_save_file', type = str)

    parser.add_argument('--cache_workers', type = int, default=1)
    parser.add_argument('--input_name', type = str)
    parser.add_argument('--cache_name', type = str, default = 'rc_fusion_graph.t5')
    parser.add_argument('--max_tabs', type = int, default = 3)
    ### Training Arguments
    parser.add_argument('--max_epoch',type=int, default = 3)
    parser.add_argument('--warm_rate', type=int,default = 0)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--batch_size", default=2,type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--print_freq",type = int,default=100)
    parser.add_argument('--seed', default = 1234, type = int)
    parser.add_argument('--fix_bert', action = 'store_true')
    parser.add_argument('--save_all_ckpt', action = 'store_true')
    parser.add_argument('--rebuild_edges', action = 'store_true')
    parser.add_argument('--num_workers', type = int, default = 3)

    ### Note:Row/Col/Cell threhold are simply for monitoring the training process. They do not participate in the validation process 
    parser.add_argument('--sent_threshold', type = float, default = 0.0)
    parser.add_argument('--col_threshold', type = float, default = 0.1)
    parser.add_argument('--row_threshold', type = float, default = 0.1)
    parser.add_argument('--cell_threshold', type = float, default = 0.04)
    parser.add_argument('--new_cache', action = 'store_true', help = 'build cache from scratch.')

    ### Model Architecture
    parser.add_argument('--edge_pattern', type = str, default = 'page_and_table', help = 'Edge Connection Pattern: full_connection / page_related / page_and_table / entity_only')
    parser.add_argument('--sent_k', type = float, default= 0.2)
    parser.add_argument('--row_k', type = float, default= 0.2)
    parser.add_argument('--col_k', type = float, default= 0.2)
    parser.add_argument('--cell_k', type = float, default= 0.2)
    return parser
