import pandas as pd
import evaluate
from simpleio import *

experiment_path = '../output/experiments/42 - Author Predictions of MSA Text'

experiment_dataframe = pd.DataFrame.from_records([
    {
        'model': job_info['model'],
        'author': job_info['author'],
        'book_name': job_info['book_name'],
        'prompt': prompt,
        'predicted_author': predicted_author,
        'ground_truth_author': ground_truth_author,
    }
    for book_path in list_dir(experiment_path, return_paths=True, default_value=[])
    for model_path in list_dir(book_path, return_paths=True, default_value=[])
    for job_info in [read_from_json_file(f'{model_path}/job.json')]
    for prompt, predicted_author, ground_truth_author in zip(
        read_from_json_file(f'{model_path}/prompts.json'),
        read_from_json_file(f'{model_path}/predicted_outputs.json'),
        read_from_json_file(f'{model_path}/ground_truth_outputs.json'),
    )
])

experiment_dataframe.to_excel(f'{experiment_path}/experiment.xlsx')
experiment_dataframe = pd.read_excel(f'{experiment_path}/experiment.xlsx')

accuracy_metric = evaluate.load('accuracy')

def compute_accuracy(model=None, author=None, book_name=None):
    filtered_dataframe = experiment_dataframe
    if model:
        filtered_dataframe = filtered_dataframe[filtered_dataframe['model'] == model]
    if author:
        filtered_dataframe = filtered_dataframe[filtered_dataframe['author'] == author]
    if book_name:
        filtered_dataframe = filtered_dataframe[filtered_dataframe['book_name'] == book_name]

    predictions = filtered_dataframe['predicted_author'].tolist()
    references = filtered_dataframe['ground_truth_author'].tolist()
    assert len(predictions) == len(references)
    return sum(prediction == reference for (prediction,reference) in zip(predictions, references)) / len(predictions)




models = set(experiment_dataframe['model'].tolist())

authors = set(experiment_dataframe['author'].tolist())
sta_per_author_dataframe = pd.DataFrame([
    {
        'model': model,
        'author': author,
        'accuracy': compute_accuracy(model=model, author=author),
    }
    for model in models
    for author in authors
])
sta_per_author_dataframe.to_excel(f'{experiment_path}/sta_per_author.xlsx')


book_names = set(experiment_dataframe['book_name'].tolist())
sta_per_author_dataframe = pd.DataFrame([
    {
        'model': model,
        'book_name': book_name,
        'accuracy': compute_accuracy(model=model, book_name=book_name),
    }
    for model in models
    for book_name in book_names
])
sta_per_author_dataframe.to_excel(f'{experiment_path}/sta_per_book.xlsx')


sta_per_model_dataframe = pd.DataFrame([
    {
        'model': model,
        'accuracy': compute_accuracy(model=model),
    }
    for model in models
])
sta_per_model_dataframe.to_excel(f'{experiment_path}/sta_per_model.xlsx')
