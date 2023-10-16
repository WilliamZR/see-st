This repo contains code for our paper Enhancing Structured Evidence Extraction for Fact Verification

##### Data

To download the dataset, run:
```sh download_data.sh```

Or you can download the data from the [FEVEROUS dataset page](https://fever.ai/dataset/feverous.html) directly. Namely:

Training Data, Development Data, Wikipedia Data as a database (sqlite3)
After downloading the data, unpack the Wikipedia data into the same folder (i.e. data).

We also provide our retrieval results at [google drive](TODO link)

##### Page Retrieval

We use the Wikipedia documents retrieved by [Hu et al., (2022)](https://aclanthology.org/2022.naacl-main.384/) for our experiments.

##### Sentence Retrieval

##### Table Retrieval

##### Cell Selection
Combine Retrieved sentences with tables
```
PYTHONPATH=src python src/my_method/cell_selection/combine_retrieved_sents_and_tabs.py --max_tabs 5 --output_file_name roberta_sent.rc_table.not_precomputed.p5.s5.t5
```
```
PYTHONPATH=src python src/my_method/cell_selection/graph_generator.py --input_name roberta_sent.rc_table.not_precomputed.p5.s5.t3 --cache_name rc_fusion_graph.t3 --new_cache --cache_workers 8 --roberta_path /home/wuzr/bert_weights/roberta-base --tapas_path /home/wuzr/bert_weights/tapas-base
```
##### Verdict Prediction

##### Evaluation and Submission
To evaluate table retrieval, run the following command. Only top 3 tables are used for computing table retrieval for fair comparison with previous baselines.
```
PYTHONPATH=src python src/baseline/retriever/eval_tab_retriever.py --split dev --max_page 5 --max_sent 5 --max_tabs 5
```

PYTHONPATH=src python src/my_method/cell_selection/train_graph_model.py --ckpt_root_dir checkpoints --entry_save_file dev.seest_fusion_results.old.jsonl --batch_size 4 --max_epoch 3 --device cuda:3 --edge_pattern page_and_table --cache_name rc_fusion_graph.t5 --lr 1e-6 --roberta_path /home/wuzr/bert_weights/roberta-base --tapas_path /home/wuzr/bert_weights/tapas-base 