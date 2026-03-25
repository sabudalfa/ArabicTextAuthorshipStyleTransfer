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

EXPERIMENT_PATH = '/hdd2/style_transfer/jrcai_authorship/output/experiments/33 - gemma-9b - generation/'

llm_loaders = [
    # LLMLoader(
    #     '/hdd/shared_models/jais-family-30b-16k-chat/',
    #     llm_initializer=JAISFamily30B16KChatInitializer(max_length=4096),
    # ),
    # LLMLoader(
    #     '/hdd/shared_models/Meta-Llama-3.1-8B-Instruct/',
    #     llm_initializer=Llama31Initializer(max_length=4096),
    # ),
    LLMLoader(
        '/hdd/shared_models/gemma-2-9b-it/',
        llm_initializer=LLMInitializer(
            max_length=3000,
        )
    ),
]

tokenizer = LLMLoader(
    '/hdd/shared_models/jais-family-13b-chat/',
    llm_initializer=JAISFamily13BChatInitializer(),
).load_tokenizer()

def get_samples(llm_name, use_fine_tuning_prompt, author=None, split=None):
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
    for llm_loader in llm_loaders
    for author in authors
    for job in [
        ExperimentDirInitializationJob(
            job_information={
                'author': author,
                'model': llm_loader.name,
                'shots_count': 0,
            },
            test_samples=get_samples(
                llm_name=llm_loader.name,
                split='test',
                author=author,
                use_fine_tuning_prompt=False,
            ),
            experiment_path=f'{EXPERIMENT_PATH}/{author}/outputs/{llm_loader.name}/zero-shot/',
        ),
        ExperimentDirInitializationJob(
            job_information={
                'author': author,
                'model': f'{llm_loader.name}-fine-tuned-using-single-author',
                'shots_count': 0,
            },
            test_samples=get_samples(
                llm_name=llm_loader.name,
                split='test',
                author=author,
                use_fine_tuning_prompt=True,
            ),
            experiment_path=f'{EXPERIMENT_PATH}/{author}/outputs/{llm_loader.name}/fine-tuned-using-single-author/',
        ),
        # ExperimentDirInitializationJob(
        #     job_information={
        #         'author': author,
        #         'model': f'{llm_loader.name}-fine-tuned-using-all-authors',
        #         'shots_count': 0,
        #     },
        #     test_samples=get_samples(
        #         llm_name=llm_loader.name,
        #         split='test',
        #         author=author,
        #         use_fine_tuning_prompt=True,
        #     ),
        #     experiment_path=f'{EXPERIMENT_PATH}/{author}/outputs/{llm_loader.name}/fine-tuned-using-all-authors/',
        # ),
    ]
])

run_llm_jobs_in_parallel([
    job
    for llm_loader in llm_loaders
    for author in authors
    for job in [
        TextGenerationJob(
            llm_loader=llm_loader,
            experiment_path=f'{EXPERIMENT_PATH}/{author}/outputs/{llm_loader.name}/zero-shot/',
            gpus_count=2,
            use_chat_format=True,
        ),
        TextGenerationJob(
            llm_loader=LLMLoader(
                model_path=f'{EXPERIMENT_PATH}/{author}/fine-tuned-models/{llm_loader.name}-for-author-style-transfer/',
                tokenizer_path=llm_loader.tokenizer_path,
                llm_initializer=llm_loader.llm_initializer,
            ),
            experiment_path=f'{EXPERIMENT_PATH}/{author}/outputs/{llm_loader.name}/fine-tuned-using-single-author/',
            gpus_count=2,
            use_chat_format=False,
        ),
        # TextGenerationJob(
        #     llm_loader=LLMLoader(
        #         model_path=f'{EXPERIMENT_PATH}/all_authors/fine-tuned-models/{llm_loader.name}-for-author-style-transfer/',
        #         tokenizer_path=llm_loader.tokenizer_path,
        #         llm_initializer=llm_loader.llm_initializer,
        #     ),
        #     experiment_path=f'{EXPERIMENT_PATH}/{author}/outputs/{llm_loader.name}/fine-tuned-using-all-authors/',
        #     gpus_count=2,
        #     use_chat_format=False,
        # ),
    ]
])

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
