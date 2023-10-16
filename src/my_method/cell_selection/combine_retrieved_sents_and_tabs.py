import jsonlines
import argparse
from utils.annotation_processor import AnnotationProcessor
from collections import defaultdict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tabs', type = int, default = 3)
    parser.add_argument('--sentence_file_name', type = str, default='sentences.roberta.p5.s5')
    parser.add_argument('--table_file_name', type = str, default = 'rc.p5.t5')
    parser.add_argument('--output_file_name', type=str, default='roberta_sent.rc_table.not_precomputed.p5.s5.t3')
    args = parser.parse_args()

    
    for split in ['train', 'dev', 'test']:
        #annotator = AnnotationProcessor('data/{}.jsonl'.format(split))
        
        sentence_path = 'data/{0}.{1}.jsonl'.format(split, args.sentence_file_name)
        sentence_data = defaultdict(list)
        with jsonlines.open(sentence_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                sentence_data[line['id']] = line['predicted_sentences']
        table_path = 'data/{0}.{1}.jsonl'.format(split, args.table_file_name)
        table_data = defaultdict(list)
        with jsonlines.open(table_path, 'r') as f:
            for line in f:
                table_data[line['id']] = line['predicted_tables'][:args.max_tabs]

        output_path = 'data/{}.{}.jsonl'.format(split, args.output_file_name)

        with jsonlines.open('data/{}.jsonl'.format(split), 'r') as f:
            annotations = [case for case in f][1:]

        with jsonlines.open(output_path, 'w') as writer:
            writer.write({'header': ''})
            for anno in annotations:
                id = anno['id']
                if 'annotator_operations' in anno.keys():
                    del anno['annotator_operations']

                predicted_evidence = sentence_data[id] + table_data[id]
                anno['predicted_evidence'] = predicted_evidence
                writer.write(anno) 

                
