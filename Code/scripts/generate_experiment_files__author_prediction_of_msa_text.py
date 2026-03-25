import pandas as pd
from llm import *
from llm_jobs import *


output_experiment_path = '../output/experiments/42 - Author Predictions of MSA Text/'

dataframe = pd.read_excel('../data/authorship_dataset_v2.xlsx')
dataframe = dataframe[dataframe['split'] == 'test']

llm_loader = LLMLoader(
    model_path='../output/experiments/30 - author-classification/models/jais-family-13b-chat-fine-tuned-for-author-classification/',
    tokenizer_path='/hdd/shared_models/jais-family-13b-chat/',
    llm_initializer=JAISFamily13BChatInitializer(
        max_new_tokens=20,
    ),
)
tokenizer = llm_loader.load_tokenizer()

def get_samples(book_name):
    book_dataframe = dataframe[dataframe['book_name'] == book_name]
    prompts = [
        f'{text_in_msa}<seperator>'
        for text_in_msa in book_dataframe['text_in_msa']
    ]
    authors = book_dataframe['author'].tolist()
    samples = zip(prompts, authors)
    return [
        (tokenizer.decode(tokenizer.encode(input_text)[:1900]), output_text)
        for input_text, output_text in samples
    ]


run_llm_jobs_in_parallel([
    ExperimentDirInitializationJob(
        job_information={'book': book_name},
        test_samples=get_samples(book_name),
        experiment_path=f'{output_experiment_path}/{book_name}/',
    )
    for book_name in dataframe['book_name'].unique()
])

run_llm_jobs_in_parallel([
    TextGenerationJob(
        llm_loader=llm_loader,
        experiment_path=f'{output_experiment_path}/{book_name}/',
        use_chat_format=False,
        gpus_count=1,
    )
    for book_name in dataframe['book_name'].unique()
])