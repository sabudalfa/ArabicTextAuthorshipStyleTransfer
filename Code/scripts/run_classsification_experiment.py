import pandas as pd
from llm import *
from llm_jobs import *

EXPERIMENT_PATH = '../output/experiments/30 - author-classification/'

dataframe = pd.read_excel('../data/authorship_dataset_v2.xlsx')
dataframe = dataframe.sample(frac=1.0)

def remove_samples_with_many_tokens(samples, tokenizer):
    filtered_samples = [
        (input_text, output_text)
        for input_text, output_text in samples
        if len(tokenizer.encode(input_text) + tokenizer.encode(output_text)) <= 1900
    ]
    print(f'Dropped samples: {len(samples) - len(filtered_samples)}/{len(samples)}')
    return filtered_samples

authors = list(set(dataframe['author'].tolist()))

llm_loaders = [
    LLMLoader(
        '/hdd/shared_models/Meta-Llama-3-8B/',
        llm_initializer=Llama31Initializer(
            do_sample=False,
        ),
    ),
    LLMLoader(
        '/hdd/shared_models/jais-family-13b-chat/',
        llm_initializer=JAISFamily13BChatInitializer(
            do_sample=False,
        ),
    ),
]


tokenizer = LLMLoader(
    '/hdd/shared_models/jais-family-13b-chat/',
    llm_initializer=JAISFamily13BChatInitializer(),
).load_tokenizer()


def get_samples(split, author=None):
    filtered_dataframe = dataframe[dataframe['split'] == split]
    if author:
        filtered_dataframe = filtered_dataframe[filtered_dataframe['author'] == author]
    return remove_samples_with_many_tokens(
        samples=[
            (row['text_in_author_style'] + '<seperator>', row['author'])
            for _, row in filtered_dataframe.iterrows()
        ],
        tokenizer=tokenizer,
    )


# run_llm_jobs_in_parallel([
#     LLMTrainingJob(
#         llm_trainer=LLMTrainer(
#             llm_loader=loader,
#             train_samples=get_samples('train'),
#             eval_samples=get_samples('eval'),
#             peft_config=LoRAConfigRepository.llama_3() if loader.name == 'Meta-Llama-3-8B' else LoRAConfigRepository.jais_v1(),
#             learning_rate=2.5e-4,
#             epochs_count=10,
#             train_batch_size=2,
#             eval_batch_size=2,
#             output_dir=f'{EXPERIMENT_PATH}/models/{loader.name}-fine-tuned-for-author-classification/'
#         ),
#         gpus_count=2,
#     )
#     for loader in llm_loaders
# ])


run_llm_jobs_in_parallel([
    ExperimentDirInitializationJob(
        job_information={
            'loader': loader.name,
        },
        test_samples=get_samples('test'),
        experiment_path=f'{EXPERIMENT_PATH}/output/{loader.name}/all-authors/'
    )
    for loader in llm_loaders
])

run_llm_jobs_in_parallel([
    TextGenerationJob(
        llm_loader=LLMLoader(
            model_path=f'{EXPERIMENT_PATH}/models/{loader.name}-fine-tuned-for-author-classification/',
            tokenizer_path=loader.tokenizer_path,
            llm_initializer=loader.llm_initializer,
        ),
        experiment_path=f'{EXPERIMENT_PATH}/output/{loader.name}/all-authors/',
        use_chat_format=False,
        gpus_count=1,
    )
    for loader in llm_loaders
])

# run_llm_jobs_in_parallel([
#     LLMEvaluationJob(
#         llm_evaluator=LLMEvaluator(
#             evaluation_metrics_getter=lambda: [
#                 BleuMetric(),
#                 ChrfMetric(),
#             ],
#         ),
#         experiment_path=f'./output/{loader.name}/',
#     )
#     for loader in llm_loaders
# ])
#
# ResultsAggregationJob(
#     output_dir='./output/',
#     job_paths=[
#         f'./output/{loader.name}/'
#         for loader in llm_loaders
#     ],
# )()
