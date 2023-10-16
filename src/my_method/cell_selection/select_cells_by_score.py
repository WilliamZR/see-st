import sys
import numpy as np
from my_utils.common_utils import load_pkl_data
import argparse
import jsonlines
from tqdm import tqdm
from collections import defaultdict
from utils.annotation_processor import AnnotationProcessor
def average(list):
    return float(sum(list) / len(list))

def select_cells(cell_score, threshold = 0.1):
    predicted_cells = [item[0] for item in cell_score if item[1] > threshold]
    if len(predicted_cells) > 25:
        cell_score.sort(key = lambda x : float(x[1]), reverse = True)
        predicted_cells = list(list(zip(*cell_score))[0][:25])
    return predicted_cells
def select_sents(sent_score, threshold = 0.1):
    predicted_sents = [item[0] for item in sent_score if np.exp(item[1][1]) > threshold]
    return predicted_sents

def get_metric_item(docs_gold, docs_predicted):
    coverage_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_gold)
    if docs_predicted:
        precision_ele = len(set(docs_predicted) & set(docs_gold)) / len(docs_predicted)
    else:
        precision_ele = 0
    return coverage_ele, precision_ele

def get_evidence_type(gold_evidence):
    ## 0 for sentence
    ## 1 for cell
    ## 2 for joint
    type_set = set()
    for evi in gold_evidence:
        if '_cell_' in evi:
            type_set.add(1)
        elif '_sentence' in evi:
            type_set.add(0)
    if len(type_set) > 1:
        return 2
    
    else:
        return type_set.pop()

def display_follow_evidence_type(data_dict):
    type_dict = {0 : 'SENTENCE', 1 : 'TABLE', 2 : 'JOINT'}
    for key in data_dict.keys():
        print(type_dict[key], end = ':')
        print(average(data_dict[key]))
    print('\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type = str, default= 'dev')
    parser.add_argument('--cell_threshold', type = float, default = 0.05)
    parser.add_argument('--sent_threshold', type = float, default= 0.05)
    ## 0.001,0.005,0.01,0.05
    args = parser.parse_args()
    print(args)
    input_file = '/home/wuzr/feverous/data/{0}.t3.rc_graph_results.jsonl'.format(args.split)

    output_file = '/home/wuzr/feverous/data/{}.rc.analysis.05.jsonl'.format(args.split)
    
    annotation_processor = AnnotationProcessor('/home/wuzr/feverous/data/{}.jsonl'.format(args.split))
    

    if args.split == 'test':
        annotation_by_id = {el.get_id(): el for el in annotation_processor}
    else:
        annotation_by_id = {el.get_id(): el for el in annotation_processor if el.has_evidence()}

    sent_selection_ratio_by_type = defaultdict(list)
    cell_selection_ratio_by_type = defaultdict(list)
    sent_selection_num_by_type = defaultdict(list)
    cell_selection_num_by_type = defaultdict(list)
    sentence_evidence_ratio_by_type = defaultdict(list)

    coverage_sents = []
    coverage_cells = []
    precision_sents = []
    precision_cells = []
    with jsonlines.open(input_file, 'r') as f:
        with jsonlines.open(output_file, 'w') as writer:
            #writer.write({'header':''})
            for idx, line in tqdm(enumerate(f)):
                evi_id = int(line['id'])
                cell_scores = line['predicted_cell_scores']
                predicted_evidence = line['predicted_evidence']
                ### find the first item contains '_cell_', return index, if no item contains '_cell_', return None
                if cell_scores:
                    cell_index = next((index for (index, d) in enumerate(predicted_evidence) if '_cell_' in d), None)
                    assert cell_index != None
                    cell_scores = np.exp([float(score) for score in cell_scores])
                    ### get all index that cell score is bigger than args.cell_threshold
                    cell_index_list = [index for (index, d) in enumerate(cell_scores) if d > args.cell_threshold]
                    ### index + first cell index in predicted_evidence
                    cell_index_list = [index + cell_index for index in cell_index_list]
                    ### get predicted_evidence
                    predicted_cells = [predicted_evidence[index] for index in cell_index_list]
                    predicted_sents = predicted_evidence[:cell_index]
                    predicted_evidence = predicted_sents + predicted_cells
                    line['predicted_evidence'] = predicted_evidence


                if args.split != 'test':
                    sents_gold = set()
                    cells_gold = set()
                    
                    for item in annotation_by_id[evi_id].get_evidence(flat = True):
                        if '_sentence_' in item:
                            sents_gold.add(item)
                        elif '_cell_' in item:
                            cells_gold.add(item)
                    if (len(sents_gold) + len(cells_gold)) > 0:
                        evidence_type = get_evidence_type(sents_gold.union(cells_gold))

                        sent_selection_num_by_type[evidence_type].append(len(predicted_sents))
                        cell_selection_num_by_type[evidence_type].append(len(predicted_cells))
                        if (len(predicted_cells) + len(predicted_sents)) > 0:
                            sentence_evidence_ratio_by_type[evidence_type].append(len(predicted_sents) / (len(predicted_cells) + len(predicted_sents)))
                    if sents_gold:
                        sent_coverage_ele, sent_precision_ele = get_metric_item(sents_gold, predicted_sents)
                        coverage_sents.append(sent_coverage_ele)
                        precision_sents.append(sent_precision_ele)
                    if cells_gold:
                        cell_coverage_ele, cell_precision_ele = get_metric_item(cells_gold, predicted_cells)
                        coverage_cells.append(cell_coverage_ele)
                        precision_cells.append(cell_precision_ele)
                    line['evidence'] = annotation_by_id[evi_id].evidence
                writer.write(line)
    if args.split != 'test':
        print("Cell Recall")
        print(average(coverage_cells))
        print('Cell Precision')
        print(average(precision_cells))

        print('Sentence Recall')
        print(average(coverage_sents))
        print('Sentence Precision')
        print(average(precision_sents))

        print('Average Statistics on Evidence Type')
        print('Sentence Selection Number')
        display_follow_evidence_type(sent_selection_num_by_type)
        print('Sentence Selection Ratio From Candiates (top5)')
        display_follow_evidence_type(sent_selection_ratio_by_type)
        print('Cell Selection Number')
        display_follow_evidence_type(cell_selection_num_by_type)

    print('Retrieval Finished')