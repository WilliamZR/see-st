cd ..
PYTHONPATH=src python src/my_method/table_extraction/prepare_rc_data.py\
    --db /home/wuzr/feverous/data/feverous_wikiv1.db\
    --data_path data&&
PYTHONPATH=src python src/my_method/table_extraction/train_rc_retriever.py\
    --device cuda:0\
    --input_path data\
    --tapas_path ~/bert_weights/tapas-base\
    --model_save_path checkpoints/\
    --max_tabs 5\
    --wiki_path /home/wuzr/feverous/data/feverous_wikiv1.db\
    --lr 1e-5\
    --weight_decay 1e-5\
    --gamma 0\
    --batch_size 8\
    --seed 229\
    --select_criterion col&&
ckpt_name=$(ls checkpoints -tr|grep RC_MLP_Retriever|tail -1) &&
PYTHONPATH=src python src/my_method/table_extraction/eval_rc_retriever.py\
    --device cuda:0\
    --model_load_path checkpoints/${ckpt_name}\
    --select_criterion col
    --max_tabs 3

