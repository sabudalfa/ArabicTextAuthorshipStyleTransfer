import pandas as pd
from transformers import *

dataframe = pd.read_excel('/hdd2/style_transfer/jrcai_authorship/data/authorship_dataset_v2.xlsx')

tokenizer = AutoTokenizer.from_pretrained('/hdd/shared_models/jais-family-13b-chat/')

tokenizer.padding_side = 'left'


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


def get_samples(use_fine_tuning_prompt, author=None, split=None):
    filtered_dataframe = dataframe.copy()
    if author and author != 'all_authors':
        filtered_dataframe = filtered_dataframe[filtered_dataframe['author'] == author]
    if split:
        filtered_dataframe = filtered_dataframe[filtered_dataframe['split'] == split]
    samples = convert_dataframe_to_samples(filtered_dataframe, use_fine_tuning_prompt)
    return remove_samples_with_many_tokens(samples, tokenizer)


authors = dataframe['author'].unique().tolist()
splits = dataframe['split'].unique().tolist()

for author in authors:
    for split in splits:
        print(f'{author}-{split} samples count =', len(get_samples(True, author, split)))
