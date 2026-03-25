import pandas as pd
from simpleio import *


all_samples = []
for dataset_path in ['../data/AuthorshipDatasetV1 - splitted/', '../data/AuthorshipDatasetV2Final - splitted']:
    for split in list_dir(dataset_path):
        split_path = f'{dataset_path}/{split}/'
        for author in list_dir(split_path):
            author_path = f'{split_path}/{author}/'
            for book_file_name in list_dir(author_path):
                book_path = f'{author_path}/{book_file_name}'
                book_dataframe = pd.read_excel(book_path)
                for text_in_msa, text_in_author_style in zip(book_dataframe['MSA'], book_dataframe['Style']):
                    all_samples.append({
                        'text_in_msa': text_in_msa,
                        'text_in_author_style': text_in_author_style,
                        'author': author,
                        'split': split,
                        'book_name': book_file_name.removesuffix('.xlsx'),
                    })


dataset_dataframe = pd.DataFrame.from_records(all_samples)
dataset_dataframe.to_excel('../data/authorship_dataset_v2.xlsx')
