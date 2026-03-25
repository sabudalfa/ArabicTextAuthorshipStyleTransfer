from llm import *

input_messages = [
    [
        {
            'role': 'user',
            'content': "What is your name?",
        }
    ],
    [
        {
            'role': 'user',
            'content': "What is the color of the sky?",
        }
    ],
]

gemma_9b_text_generator = TextGenerator.from_llm_loader(LLMLoader(
    '/hdd/shared_models/gemma-2-9b-it/',
    llm_initializer=Gemma2Initializer(
        max_length=3000,
    )
))
gemma_9b_text_generator.tokenizer.padding_side = "left"
gemma_27b_text_generator = TextGenerator.from_llm_loader(LLMLoader(
    '/hdd/shared_models/gemma-2-27b-it/',
    llm_initializer=Gemma2Initializer(
        max_length=3000,
    )
))
gemma_27b_text_generator.tokenizer.padding_side = "right"

print(gemma_9b_text_generator.generate_chat_message(input_messages, max_new_tokens=3))
print(gemma_27b_text_generator.generate_chat_message(input_messages, max_new_tokens=3))
