import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import torch
import time
import argparse
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-data_path", "--data_path", type=str, help='./public_data/DE-German/de_test.conll')
    parser.add_argument("-word_max_length", "--word_max_length", type=int, help = 4)
    parser.add_argument("-model", "--model", type=str, help='mbart')
    parser.add_argument("-model_path", "--model_name", type=str, help='mbart')
    parser.add_argument("-output_path", "--output_path", type=str, help='./de.pred.conll')

    args = parser.parse_args()

    if args.model == 'mbart':
        tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        model = MBartForConditionalGeneration.from_pretrained(args.model_path)
    else:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        model = BartForConditionalGeneration.from_pretrained(args.model_path)

    model.eval()
    model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoder = model.get_encoder()
    decoder = model.get_decoder()

    score_list, examples = [], []
    guid_index = 1
    with open(args.data_path, "r", encoding="utf-8") as f:
        words, labels = [], []
        for line in f:
            if line.startswith("-DOCSTART-") or line.startswith('#') or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(words=words, labels=labels))
                    words, labels = [], []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(words=words, labels=labels))
                    
            
    trues_list, preds_list = [], []
    num_01 = len(examples)
    num_point = 0
    start = time.time()
    for example in examples:
        sources = ' '.join(example.words)
        preds_list.append(prediction(sources))
        trues_list.append(example.labels)
        if num_point % 100 == 0:
            print('%d/%d (%s)'%(num_point+1, num_01, cal_time(start)))
            print(example.words)
            print('Pred:', preds_list[num_point])
        num_point += 1
        

    for num_point in range(len(preds_list)):
        preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'

    final_preds = []
    for x in preds_list:
        final_preds.extend([' '] + x.split() + [' '])
    pd.DataFrame(final_preds).to_csv(args.output_path,  header=None,  index=False)
