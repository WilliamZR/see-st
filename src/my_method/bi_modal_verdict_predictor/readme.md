# Bi Modal

### Training
To train the verdict prediction model run respectively:
```
PYTHONPATH=src python src/my_methods/bi_modal_verdict_predictor/train_verdict_predictor.py --model_type BiModalCls --data_dir data --train_data_type both
PYTHONPATH=src python src/my_methods/bi_modal_verdict_predictor/train_verdict_predictor.py --model_type BiModalClsConcat --data_dir data --train_data_type both
PYTHONPATH=src python src/my_methods/bi_modal_verdict_predictor/train_verdict_predictor.py --model_type WeightedBiModalCls --data_dir data --train_data_type both

```

### Verdict Prediction
To predict the verdict
```
PYTHONPATH=src python src/my_methods/bi_modal_verdict_predictor/eval_verdict_predictor.py --data_dir data
 ```

### Evaluation
To evaluate your generated predictions locally, simply run the file `evaluate.py` as following:
```
python evaluation/evaluate.py --input_path data/dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl
 ```

#### Use retrieval results from Fusion Graph
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src nohup python src/my_methods/bi_modal_verdict_predictor/train_verdict_predictor.py --model_type BiModalCls --train_data_type both --data_dir data/fusion_graph_data  > fusion_graph_train_1013.log &