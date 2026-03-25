import uuid
from typing import *
import pandas as pd
from torch.utils.data import DataLoader
from transformers import *
from datasets import Dataset
import random
import evaluate


EXPERIMENT_PATH = '../output/experiments/31 - author-classification with roberta/'

dataframe = pd.read_excel('../data/authorship_dataset_v2.xlsx')


authors = list(set(dataframe['author'].tolist()))[:10]
dataframe = pd.concat([
    dataframe[(dataframe['author'] == author) & (dataframe['split'] == split)][:400]
    for author in authors
    for split in ['train', 'eval', 'test']
])

label2id = {
    author: idx
    for idx, author in enumerate(authors)
}

MAX_SAMPLES = 2 ** 32

train_dataframe = dataframe[dataframe['split'] == 'train'][:MAX_SAMPLES]
train_samples = [
    (row['text_in_author_style'], label2id[row['author']])
    for _, row in train_dataframe.iterrows()
]
random.shuffle(train_samples)

eval_dataframe = dataframe[dataframe['split'] == 'eval'][:MAX_SAMPLES]
eval_samples = [
    (row['text_in_author_style'], label2id[row['author']])
    for _, row in eval_dataframe.iterrows()
]
random.shuffle(eval_samples)

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
)


def to_classification_dataset(samples: List[Tuple[str, int]]) -> Dataset:
    mapped_samples = []
    for text, label in samples:
        mapped_samples.append({
            'input_ids': tokenizer.encode(text, truncation=True),
            'label': label,
        })
    return Dataset.from_list(mapped_samples)



accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_prediction: EvalPrediction):
    predictions = eval_prediction.predictions.argmax(axis=1)
    references = eval_prediction.label_ids
    return accuracy_metric.compute(references=references, predictions=predictions)


records = []
for lr in [5e-3, 2e-4, 1e-3, 7.5e-4, 5e-4, 1e-4]:
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=f'{EXPERIMENT_PATH}/models/{uuid.uuid4()}',
            do_eval=True,
            eval_steps=1,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=lr,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            disable_tqdm=False,
            weight_decay=0.01,
            remove_unused_columns=False,
            bf16=True,
            report_to=None,
            load_best_model_at_end=True,
        ),
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=to_classification_dataset(train_samples),
        eval_dataset=to_classification_dataset(eval_samples),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )
    trainer.train()
    records.append({
        'lr': lr,
        'eval': trainer.evaluate(to_classification_dataset(eval_samples)),
    })
    print(records)

