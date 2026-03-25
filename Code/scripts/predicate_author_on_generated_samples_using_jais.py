from llm import *
from llm import JAISFamily13BChatInitializer
from llm_jobs import *
from simpleio import *

paths = [
    model_path
    for book_path in list_dir('../output/experiments/38 - AceGPT-v2-8B-Chat - by book/', return_paths=True)
    for model_path in list_dir(book_path, return_paths=True, default_value=[])
]

llm_loader = LLMLoader(
    model_path='../output/experiments/30 - author-classification/models/jais-family-13b-chat-fine-tuned-for-author-classification/',
    tokenizer_path='/hdd/shared_models/jais-family-13b-chat/',
    llm_initializer=JAISFamily13BChatInitializer(
        max_new_tokens=20,
    ),
)
tokenizer = llm_loader.load_tokenizer()


def get_samples(path: str):
    texts = read_from_json_file(f'{path}/predicted_outputs.json')
    texts = [
        text + '<seperator>'
        for text in texts
    ]
    filtered_texts = tokenizer.batch_decode([
        tokenizer.encode(text)[:1900]
        for text in texts
    ])
    print(sum(a == b for a, b in zip(texts, filtered_texts)), '/', len(filtered_texts))
    texts = filtered_texts
    authors = len(texts) * [read_from_json_file(f'{path}/job.json')['author']]
    samples = list(zip(texts, authors))
    return [
        (tokenizer.decode(tokenizer.encode(input_text)[:1900]), output_text)
        for input_text, output_text in samples
    ]


run_llm_jobs_in_parallel([
    ExperimentDirInitializationJob(
        job_information=read_from_json_file(
            file_path=f'{path}/job.json'
        ),
        test_samples=get_samples(path),
        experiment_path=path.replace(
            '38 - AceGPT-v2-8B-Chat - by book',
            '38 - AceGPT-v2-8B-Chat - by book - STA',
        ),
    )
    for path in paths
])

run_llm_jobs_in_parallel([
    TextGenerationJob(
        llm_loader=llm_loader,
        experiment_path=path.replace(
            '38 - AceGPT-v2-8B-Chat - by book',
            '38 - AceGPT-v2-8B-Chat - by book - STA',
        ),
        use_chat_format=False,
        gpus_count=1,
    )
    for path in paths
])
