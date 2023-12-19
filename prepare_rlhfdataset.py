import pandas as pd
from datasets import Dataset

file_path = 'clang8_source_target_en.spacy_tokenized.tsv'
data = pd.read_csv(file_path , sep='\t', encoding='UTF-8',header = None, quotechar="\0")
data.columns = ['source', 'target']

sources = []
targets = []
formatted_strings = []
for index, row in data.iterrows():
    if row["target"] is None:
        continue
    if row["target"] == row["source"]:
        continue
    formatted_string = f"GEC: {row['source']}\nCorrected:"
    formatted_strings.append(formatted_string)
    sources.append(row['source'])
    targets.append(row['target'])

dataset = Dataset.from_dict({'text': formatted_strings, 'target': targets, 'source': sources})
print(dataset)
dataset.save_to_disk('rlhfdataset')