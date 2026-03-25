from llm import *
import pandas as pd


dataframe = pd.read_excel('../data/authorship_dataset_v2.xlsx')
print(dataframe.columns)
sample_text = dataframe.iloc[0]['text_in_msa']

print(sample_text)

acegpt_loader = LLMLoader(
    '/hdd/shared_models/AceGPT-13B-chat/',
    llm_initializer=Llama2Initializer(),
)

jais_loader = LLMLoader(
    '/hdd/shared_models/jais-family-13b-chat/',
    llm_initializer=JAISFamily13BChatInitializer(),
)

llama_loader = LLMLoader(
    '/hdd/shared_models/Meta-Llama-3-8B/',
    llm_initializer=Llama31Initializer(),
)

gemma_loader = LLMLoader(
    '/hdd/shared_models/gemma-2-9b',
)

acegpt_tokenizer = acegpt_loader.load_tokenizer()
jais_tokenizer = jais_loader.load_tokenizer()
llama_tokenizer = llama_loader.load_tokenizer()
gemma_tokenizer = gemma_loader.load_tokenizer()

print(f'AceGPT number of tokens = {len(acegpt_tokenizer.encode(sample_text))}')
print(f'Jais number of tokens = {len(jais_tokenizer.encode(sample_text))}')
print(f'Llama number of tokens = {len(llama_tokenizer.encode(sample_text))}')
print(f'Gemma number of tokens = {len(gemma_tokenizer.encode(sample_text))}')

for text_in_style, text_in_author_style in zip(dataframe['text_in_style'], dataframe['text_in_author_style']):
    if len(jais_tokenizer.encode(text_in_style + text_in_author_style)) > 2048:
        print(text_in_style)
        print(text_in_author_style)
        break
