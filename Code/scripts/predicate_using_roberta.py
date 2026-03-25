from typing import *
import pandas as pd
from transformers import *
from datasets import Dataset
import torch
import tqdm

from simpleio import write_to_json_file

dataframe = pd.read_excel('../data/authorship_dataset_v2.xlsx')

authors = sorted(list(set(dataframe['author'].tolist())))

label2id = {
    author: idx
    for idx, author in enumerate(authors)
}

MAX_SAMPLES = 2 ** 32


test_dataframe = dataframe[dataframe['split'] == 'test'][:MAX_SAMPLES]
test_samples = [
    (row['text_in_author_style'], label2id[row['author']])
    for _, row in test_dataframe.iterrows()
]

model_id = 'xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(authors),
    id2label={
        i: author
        for i, author in enumerate(authors)
    },
    label2id=label2id,
    device_map='cuda:0',
)

def to_classification_dataset(samples: List[Tuple[str, int]]) -> Dataset:
    mapped_samples = []
    for text, label in samples:
        mapped_samples.append({
            'input_ids': tokenizer.encode(text, truncation=True),
            'label': label,
        })
    return Dataset.from_list(mapped_samples)


model.eval()
text_list = []
ground_truth_authors = []
predicted_authors = []
for text, actual_label_id in tqdm.tqdm(test_samples):
    model_inputs = {'input_ids': torch.tensor([tokenizer.encode(text, truncation=True)], device='cuda')}
    predicted_y = model(**model_inputs).logits.argmax(dim=1)
    print(predicted_y)

    text_list.append(text)
    ground_truth_authors.append(authors[actual_label_id])
    predicted_authors.append(authors[predicted_y])
exit(0)

write_to_json_file(
    file_path='../output/experiments/31 - author-classification with roberta/text_list.json',
    content=text_list,
)
write_to_json_file(
    file_path='../output/experiments/31 - author-classification with roberta/ground_truth_authors.json',
    content=ground_truth_authors,
)
write_to_json_file(
    file_path='../output/experiments/31 - author-classification with roberta/predicted_authors.json',
    content=predicted_authors,
)