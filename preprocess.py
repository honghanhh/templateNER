import os
import glob
import argparse
import pandas as pd
from utils_metrics import *
import warnings
warnings.filterwarnings("ignore")

def template_format_with_non_entity(data_path, lang='en'):
    df = read_conll(data_path)
    df['input_text'] = [' '.join(x).strip() for x in df['tokens']]
    df['entities'] = [get_entities(x) for x in df['spans']]
    df['len'] = [len(x) for x in df.entities]    
    df = df[df['len'] != 0]
    vowels = ('a','e','i','o','u','A','E','I','O','U')
    df['target_text'] = pd.Series(dtype='object')
    df['texts_wo_nes'] = df['input_text'].copy()
    for i in range(len(df)):
        li = []
        # This is for template #2, feel free to adapt to the template you want
        for x in df.entities.iloc[i]:
            text = ' '.join(df.tokens.iloc[i][x[1]:x[2]+1])
            if lang == 'en':
                li.append(text + ' belongs to ' + x[0] + ' category')
            elif lang == 'de':
                li.append(text + ' gehört zur Kategorie ' + x[0])
            elif lang == 'zh':
                li.append(text + ' 屬於 ' + x[0] +' 類')
            elif lang == 'hi':
                li.append(text + ' ' + x[0] +' से संबंधित है')
            elif lang == 'bn':
                li.append(text + ' ' + x[0] +' এর অন্তর্গত') 
            else: #template en #1 by default
                if x[0].startswith(vowels):
                    li.append(text + ' is an ' + x[0] + ' entity')
                else:
                    li.append(text + ' is a ' + x[0] + ' entity') 
            df['texts_wo_nes'].iloc[i] = df['texts_wo_nes'].iloc[i].replace(text, '')
        df['target_text'].iloc[i] = li
    if lang == 'en':
        df['token_wo_nes'] = [[y + ' does not belong to any category' for y in x.split()] for x in df['texts_wo_nes']]
    elif lang == 'de':
        df['token_wo_nes'] = [[y + ' gehört zu keiner Kategorie' for y in x.split()] for x in df['texts_wo_nes']]
    elif lang == 'zh':
        df['token_wo_nes'] = [[y + ' 不屬於任何類別'  for y in x.split()] for x in df['texts_wo_nes']]
    elif lang == 'hi':
        df['token_wo_nes'] = [[y + ' किसी भी श्रेणी से संबंधित नहीं है'  for y in x.split()] for x in df['texts_wo_nes']]
    elif lang == 'bn':
        df['token_wo_nes'] = [[y + ' কোনো বিভাগের অন্তর্গত নয়৷'  for y in x.split()] for x in df['texts_wo_nes']]
    else:
        df['token_wo_nes'] = [[y + ' is not an entity' for y in x.split()] for x in df['texts_wo_nes']]
    df['target_text'] += df['token_wo_nes']
    return df[['input_text','target_text']].explode('target_text')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-input_dir", "--input_dir", type=str, help='/home/tranthh/semeval2023/train_dev')
    parser.add_argument("-lang", "--lang", type=str, help='en')
    parser.add_argument("-output_dir", "--output_dir",type=str, help='/home/tranthh/semeval2023/templated2/')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_files = glob.glob(os.path.join(args.input_dir , "*.conll"))

    li = []

    for filename in all_files:
        name = filename.split('/')[-1].split('.')[0]
        if 'test' not in name:
            df = template_format_with_non_entity(filename, args.lang)
            print(name, len(df))
            df.to_csv(args.output_dir+ name+'.csv', index=False)
