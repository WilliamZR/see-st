cd ..
PYTHONPATH=src python src/my_method/cell_selection/combine_retrieved_sents_and_tabs.py\
    --output_file_name roberta_sent.rc_table.not_precomputed.p5.s5.t5\
    --max_tabs 5 &&
PYTHONPATH=src python src/my_method/cell_selection/train_graph_model.py\
    --ckpt_root_dir checkpoints\
    --entry_save_file dev.seest_fusion_results.jsonl\
    --batch_size 4\
    --max_epoch 3\
    --device cuda:3\
    --edge_pattern page_and_table\
    --cache_name rc_fusion_graph.t5\
    --lr 1e-6\
    --roberta_path /home/wuzr/bert_weights/roberta-base\
    --tapas_path /home/wuzr/bert_weights/tapas-base &&
PYTHONPATH=src python src/my_method/cell_selection/rewrite_result.py --split dev --input_file dev.seest_fusion_results.jsonl 