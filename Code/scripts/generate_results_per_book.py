import pandas as pd
from llm import *
from llm_jobs import *
from simpleio import write_to_json_file
from tqdm import tqdm

INPUT_PATH = '../output/experiments/40 - allam'
OUTPUT_PATH = '../output/experiments/40 - allam - by book'

dataset_dataframe = pd.read_excel('../data/authorship_dataset_v2.xlsx')
dataset_dataframe = dataset_dataframe[dataset_dataframe['split'] == 'test']
book_dict = {
    row['text_in_author_style']: row['book_name']
    for _, row in dataset_dataframe.iterrows()
}

# dataframe = pd.concat([
#     pd.read_excel('../output/experiments/27 - jais-family-13b-chat/experiment.xlsx'),
#     pd.read_excel('../output/experiments/26 - llama-3.1-8b-instruct/experiment.xlsx'),
# ])
dataframe = pd.read_excel(f'{INPUT_PATH}/experiment.xlsx')
dataframe = dataframe.fillna("")
dataframe['book_name'] = [
    book_dict[row['ground_truth_output']]
    for _, row in tqdm(dataframe.iterrows())
]

job_paths = []
for book_name in set(dataframe['book_name'].tolist()):
    for model in set(dataframe['model'].tolist()):
        experiment_path = f'{OUTPUT_PATH}/{book_name}/{model}/'
        job_paths.append(experiment_path)
        filtered_dataframe = dataframe[(dataframe['book_name'] == book_name) & (dataframe['model'] == model)]
        author = list(set(filtered_dataframe['author'].tolist()))
        assert len(author) == 1
        author = author[0]
        write_to_json_file(
            file_path=f'{experiment_path}/job.json',
            content={
                'author': author,
                'model': model,
                'book_name': book_name,
            },
        )
        write_to_json_file(
            file_path=f'{experiment_path}/prompts.json',
            content=filtered_dataframe['prompt'].tolist(),
        )
        write_to_json_file(
            file_path=f'{experiment_path}/ground_truth_outputs.json',
            content=filtered_dataframe['ground_truth_output'].tolist(),
        )
        write_to_json_file(
            file_path=f'{experiment_path}/predicted_outputs.json',
            content=filtered_dataframe['predicted_output'].tolist(),
        )

run_llm_jobs_in_parallel([
    LLMEvaluationJob(
        llm_evaluator=LLMEvaluator(
            evaluation_metrics_getter=lambda: [
                BleuMetric(),
                CometMetric(),
                BertScoreMetric(lang='ar'),
                ChrfMetric(),
            ],
        ),
        experiment_path=experiment_path,
    )
    for experiment_path in job_paths
])
ResultsAggregationJob(
    output_dir=OUTPUT_PATH,
    job_paths=job_paths,
)()