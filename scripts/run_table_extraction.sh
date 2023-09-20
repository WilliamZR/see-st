cd ..
PYTHONPATH=src python src/my_method/table_extraction/prepare_rc_data.py\
    --db /home/hunan/feverous/mycode/data/feverous_wikiv1.db\
    --force_generate\
    --data_path data&&
PYTHONPATH=src python src/my_method/table_extraction/train_rc_retriever.py\
    --device cuda:1\
    --input_path data\
    --tapas_path ~/bert_weights/tapas-base\
    --model_save_path checkpoints/\
    --max_tabs 5\
    --wiki_path /home/hunan//feverous/mycode/data/feverous_wikiv1.db\
    --lr 1e-5\
    --batch_size 8


