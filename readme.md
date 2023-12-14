This repo contains code for our paper Enhancing Structured Evidence Extraction for Fact Verification at EMNLP2023

### Data

To download the dataset, run:
```sh download_data.sh```

Or you can download the data from the [FEVEROUS dataset page](https://fever.ai/dataset/feverous.html) directly. Namely:

Training Data, Development Data, Wikipedia Data as a database (sqlite3)
After downloading the data, unpack the Wikipedia data into the same folder (i.e. data).

We also provide our retrieval results at [google drive](TODO link)

### Page Retrieval and Sentence Extraction

We use the Wikipedia documents retrieved by [Hu et al., (2022)](https://aclanthology.org/2022.naacl-main.384/) for our experiments. You can download the results of page retrieval and sentence extraction from [Unifee](https://github.com/WilliamZR/unifee)


### Table Retrieval
```
cd scripts
sh run_table_extraction.sh
```

### Cell Selection
```
cd scripts
sh run_cell_selection.sh
```

### Evaluation
To evaluate table retrieval, run the following command. Only top 3 tables are used for computing table retrieval for fair comparison with previous baselines.
```
PYTHONPATH=src python src/baseline/retriever/eval_tab_retriever.py --split dev --max_page 5 --max_sent 5 --max_tabs 5
```

To evaluate the cell selection, run the following command. Only top 25 cells are used for computing table retrieval for fair comparison with previous baselines.
We make sure only cells from at most 3 tables are included during our selection code. 
```
PYTHONPATH=src python src/baseline/retriever/eval_cell_retriever.py --split dev 
```

### Contact
If you have any questions, please contact Zirui Wu (ziruiwu@pku.edu.cn)