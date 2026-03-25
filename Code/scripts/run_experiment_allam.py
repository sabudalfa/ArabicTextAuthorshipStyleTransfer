import os
import pandas as pd
from llm import *
from llm_jobs import *
from llm_jobs.llm_jobs_runner import run_llm_jobs_in_parallel
from simpleio import *


def generate_prompt(text_in_msa, author, use_fine_tuning_prompt):
    prompt = f"أعد كتابة النص التالي باسلوب {author} الأدبي.\n النص: {text_in_msa}"
    if use_fine_tuning_prompt:
        prompt += "<Seperator>"
    return prompt


def convert_dataframe_to_samples(dataframe: pd.DataFrame, use_fine_tuning_prompt):
    return [
        (
            generate_prompt(
                text_in_msa=row['text_in_msa'],
                author=row.author,
                use_fine_tuning_prompt=use_fine_tuning_prompt,
            ),
            row['text_in_author_style'],
        )
        for _, row in dataframe.iterrows()
    ]


def remove_samples_with_many_tokens(samples, tokenizer):
    filtered_samples = [
        (input_text, output_text)
        for input_text, output_text in samples
        if len(tokenizer.encode(input_text) + tokenizer.encode(output_text)) <= 1900
    ]
    print(f'Dropped samples: {len(samples) - len(filtered_samples)}/{len(samples)}')
    return filtered_samples


dataframe = pd.read_excel('/hdd2/style_transfer/jrcai_authorship/data/authorship_dataset_v2.xlsx')

EXPERIMENT_PATH = '/hdd2/style_transfer/jrcai_authorship/output/experiments/40 - allam/'

allam_api_key = read_from_file('/home/raed_mughaus/PycharmProjects/jrcai_red_teaming/allam_api_key')
message_generator = MessageGeneratorFromAllam(api_key=allam_api_key)

tokenizer = LLMLoader(
    '/hdd/shared_models/jais-family-13b-chat/',
    llm_initializer=JAISFamily13BChatInitializer(),
).load_tokenizer()

def get_samples(use_fine_tuning_prompt, author=None, split=None):
    filtered_dataframe = dataframe.copy()
    if author and author != 'all_authors':
        filtered_dataframe = filtered_dataframe[filtered_dataframe['author'] == author]
    if split:
        filtered_dataframe = filtered_dataframe[filtered_dataframe['split'] == split]
    samples = convert_dataframe_to_samples(filtered_dataframe, use_fine_tuning_prompt)
    return remove_samples_with_many_tokens(samples, tokenizer)

authors = dataframe['author'].unique().tolist()

run_llm_jobs_in_parallel([
    job
    for author in authors
    for job in [
        ExperimentDirInitializationJob(
            job_information={
                'author': author,
                'model': message_generator.name,
                'shots_count': 0,
            },
            test_samples=get_samples(
                split='test',
                author=author,
                use_fine_tuning_prompt=False,
            ),
            experiment_path=f'{EXPERIMENT_PATH}/{author}/outputs/{message_generator.name}/zero-shot/',
        ),
    ]
])

for author in authors:
    sub_experiment_path = f'{EXPERIMENT_PATH}/{author}/outputs/{message_generator.name}/zero-shot/'
    predicted_output_path = f'{sub_experiment_path}/predicted_outputs.json'
    if os.path.exists(predicted_output_path):
        continue
    write_to_json_file(
        file_path=predicted_output_path,
        content=message_generator([
            [
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
            for prompt in read_from_json_file(f'{sub_experiment_path}/prompts.json')
        ])
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
    for author in authors
    for llm_path in list_dir(f'{EXPERIMENT_PATH}/{author}/outputs/', return_paths=True)
    for experiment_path in list_dir(llm_path, return_paths=True)
    if os.path.exists(f'{experiment_path}/predicted_outputs.json')
])

ResultsAggregationJob(
    output_dir=f'{EXPERIMENT_PATH}/',
    job_paths=[
        experiment_path
        for author in authors
        for llm_path in list_dir(f'{EXPERIMENT_PATH}/{author}/outputs/', return_paths=True)
        for experiment_path in list_dir(llm_path, return_paths=True)
        if os.path.exists(f'{experiment_path}/evaluation.json')
    ],
)()
