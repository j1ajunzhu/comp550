import pandas as pd
from datasets import Dataset

file_path = 'clang8_source_target_en.spacy_tokenized.tsv'
data = pd.read_csv(file_path , sep='\t', encoding='UTF-8',header = None, quotechar="\0")
data.columns = ['source', 'target']

formatted_strings = []
for index, row in data.iterrows():
    if row["target"] is None:
        row["target"] = row["source"]
    formatted_string = f"<s> GEC: {row['source']}\nCorrected: {row['target']} </s>"
    formatted_strings.append(formatted_string)

dataset = Dataset.from_dict({'text': formatted_strings})
dataset.save_to_disk('clang8-nopacking')