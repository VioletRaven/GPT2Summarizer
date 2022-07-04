import pandas as pd
from pre_processing import cleaner_d2v
import json
import os
import argparse
import time

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_csv", type=str, required = True, help="Path to CSV\nExpecting 2 columns where 1st column --> text and 2nd column --> summary")
    parser.add_argument("--path_directory", type=str, required = True, help="Path to directory where to save .json files")
    parser.add_argument("--model", default='GroNLP/gpt2-small-italian', type=str, required=False, help="Model used for tokenizer")

    args = parser.parse_args()
    path_csv = args.path_csv
    path_directory = args.path_directory
    model = args.model

    return path_csv, path_directory, model

path_csv, path_directory, model = parser()

'''
Write to Json tokenized txt
'''

def write_json(text, summary, number, directory = path_directory):
	# saves json files
    file = os.path.join(directory, 'file_' + str(number) + '.json')
    js_example = {'id': number, 'article': text, 'abstract': summary}
    with open(file, 'w') as f:
        json.dump(js_example, f, ensure_ascii=False)


def tokenizer_to_json(dataset, model = model, directory = path_directory):
    tokenizer = cleaner_d2v.add_special_tokens(model)
    train_ids = []
    i = 0
    for index, row in dataset.iterrows():
        article, abstract = tokenizer.encode(row['text']), tokenizer.encode(row['summary'])
        if len(article) > 0 and len(abstract) > 0 and (len(article) + len(abstract)) <= 1023:
        	train_ids.append(i)
        	write_json(text = article, summary = abstract, number = i)
        i += 1
        if i % 1000 == 0:
            print(i, " files written")

    file = os.path.join(directory, 'index_files.json')

    x, y = int(len(train_ids) * 0.8), int(len(train_ids) * 0.9)
    valid_ids = train_ids[x:y]
    test_ids = train_ids[y:]
    train_ids = train_ids[:x]
    with open(file, 'w') as f:
        js = {}
        js['train_ids'] = train_ids
        js['valid_ids'] = valid_ids
        js['test_ids'] = test_ids
        json.dump(js, f)

data = pd.read_csv(path_csv)
data.drop('Unnamed: 0', inplace = True, axis = 1)
data.columns = ['text', 'summary']

print('Creating dataset...')

start = time.time()
tokenizer_to_json(dataset = data)
print('It took {} seconds to tokenize and write all .json files!'.format(time.time() - start))
