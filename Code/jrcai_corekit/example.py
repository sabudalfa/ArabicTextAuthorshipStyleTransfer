import random
from llm import *
from llm_jobs import *

words = [
    "apple", "computer", "mountain", "river",
    "language", "friendship", "adventure", "knowledge",
    "music", "sunshine"
]


def generate_tuple():
    frequent_word = random.choice(words)
    other_words = random.choices([w for w in words if w != frequent_word], k=7)
    final_words = other_words + [frequent_word] * 3
    random.shuffle(final_words)
    # word_string = ' '.join(final_words)
    # most_frequent_word = Counter(final_words).most_common(1)[0][0]
    # return (word_string, most_frequent_word)
    return ' '.join(final_words), ' '.join(reversed(final_words))


train_samples = [
    generate_tuple()
    for _ in range(160)
]
eval_samples = [
    generate_tuple()
    for _ in range(40)
]
test_samples = [
    generate_tuple()
    for _ in range(40)
]

llm_loaders = [
    LLMLoader(
        '/hdd/shared_models/Meta-Llama-3-8B/',
        llm_initializer=Llama31Initializer(),
    ),
    LLMLoader(
        '/hdd/shared_models/jais-family-13b-chat/',
        llm_initializer=JAISFamily13BChatInitializer(),
    )
]

run_llm_jobs_in_parallel([
    LLMTrainingJob(
        llm_trainer=LLMTrainer(
            llm_loader=loader,
            train_samples=train_samples,
            eval_samples=eval_samples,
            peft_config=LoRAConfigRepository.llama_3() if loader.name == 'Meta-Llama-3-8B' else LoRAConfigRepository.jais_v1(),
            learning_rate=2.5e-4,
            epochs_count=10,
            train_batch_size=16,
            eval_batch_size=16,
            output_dir=f'./models/{loader.name}-fine-tuned-for-the-great-task/'
        ),
        gpus_count=2,
    )
    for loader in llm_loaders
])

run_llm_jobs_in_parallel([
    ExperimentDirInitializationJob(
        job_information={
            'loader': loader.name,
        },
        test_samples=test_samples,
        experiment_path=f'./output/{loader.name}/'
    )
    for loader in llm_loaders
])

run_llm_jobs_in_parallel([
    TextGenerationJob(
        llm_loader=LLMLoader(
            model_path=f'./models/{loader.name}-fine-tuned-for-the-great-task/',
            tokenizer_path=loader.tokenizer_path,
            llm_initializer=loader.llm_initializer,
        ),
        experiment_path=f'./output/{loader.name}/',
        use_chat_format=False,
        gpus_count=1,
    )
    for loader in llm_loaders
])

run_llm_jobs_in_parallel([
    LLMEvaluationJob(
        llm_evaluator=LLMEvaluator(
            evaluation_metrics_getter=lambda: [
                BleuMetric(),
                ChrfMetric(),
            ],
        ),
        experiment_path=f'./output/{loader.name}/',
    )
    for loader in llm_loaders
])

ResultsAggregationJob(
    output_dir='./output/',
    job_paths=[
        f'./output/{loader.name}/'
        for loader in llm_loaders
    ],
)()

# text_generator = TextGenerator.from_llm_loader(
#     loader=LLMLoader(
#     	'./models/the-great-model/',
#         tokenizer_path=llm_loader.tokenizer_path,
#         llm_initializer=llm_loader.llm_initializer,
#     ),
#     max_samples_per_batch=1,
# )
# sentence, most_frequent_word = generate_tuple()
# print(sentence)
# print(most_frequent_word)
# print(text_generator.generate([sentence]))


# model, tokenizer, generation_config = llm_loader()
# train_llm(
#     model=model,
#     tokenizer=tokenizer,
#     train_samples=train_samples,
#     eval_samples=eval_samples,
#     peft_config=LoRAConfigRepository.llama_3(),
#     learning_rate=2.5e-4,
#     epochs_count=10,
#     train_batch_size=16,
#     eval_batch_size=16,
#     output_dir='./models/the-great-model/'
# )
