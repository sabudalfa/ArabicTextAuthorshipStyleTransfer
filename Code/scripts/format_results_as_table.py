import pandas as pd
from simpleio import *

model_name_dict = {
    'Meta-Llama-3.1-8B-Instruct': 'Meta-Llama-3.1-8B-Instruct',
    'Meta-Llama-3.1-8B-Instruct-fine-tuned-using-single-author': 'Meta-Llama-3.1-8B-Instruct',
    'Meta-Llama-3.1-8B-Instruct-fine-tuned-using-all-authors': 'Meta-Llama-3.1-8B-Instruct',
    'AceGPT-v2-8B-Chat': 'AceGPT-v2-8B-Chat',
    'AceGPT-v2-8B-Chat-fine-tuned-using-single-author': 'AceGPT-v2-8B-Chat',
    'AceGPT-v2-8B-Chat-fine-tuned-using-all-authors': 'AceGPT-v2-8B-Chat',
    'jais-family-13b-chat': 'jais-family-13b-chat',
    'jais-family-13b-chat-fine-tuned': 'jais-family-13b-chat',
    'jais-13b-fine-tuned-using-all-authors': 'jais-family-13b-chat',
    'allam': 'allam',
}

# path = '/home/raed_mughaus/Files/Authorship Results/21 authors - Meta-Llama-3.1-8B-Instruct/summary.xlsx'
# path = '/home/raed_mughaus/Files/Authorship Results/21 authors - AceGPT-v2-8B-Chat/summary.xlsx'
# path = '/home/raed_mughaus/Files/Authorship Results/21 authors - jais-family-13b-chat/summary.xlsx'
# path = '/home/raed_mughaus/Files/Authorship Results/21 authors - allam/summary.xlsx'

# path = '/home/raed_mughaus/Files/Authorship Results/Author Classification/STA of allam/sta_per_author.xlsx'
# path = '/home/raed_mughaus/Files/Authorship Results/Author Classification/STA of Jais and Llama/sta_per_author.xlsx'
# path = '/home/raed_mughaus/Files/Authorship Results/Author Classification/STA of AceGPT-v2-8B-Chat/sta_per_author.xlsx'

dataframe = pd.concat([
    pd.read_excel(path)
    for path in [
        '/home/raed_mughaus/Files/Authorship Results/21 authors - Meta-Llama-3.1-8B-Instruct/summary.xlsx',
        '/home/raed_mughaus/Files/Authorship Results/21 authors - AceGPT-v2-8B-Chat/summary.xlsx',
        '/home/raed_mughaus/Files/Authorship Results/21 authors - jais-family-13b-chat/summary.xlsx',
        '/home/raed_mughaus/Files/Authorship Results/21 authors - allam/summary.xlsx',
    ]
])

authors = sorted(dataframe['author'].unique().tolist())
models = sorted(dataframe['model'].unique().tolist())

def get_score(model, author, metric_name):
    x = dataframe[(dataframe['model'] == model) & (dataframe['author'] == author)][metric_name].tolist()
    assert len(x) == 1
    return x[0]


def get_accuracy(model, author):
    x = sta_dataframe[(sta_dataframe['model'] == model) & (sta_dataframe['author'] == author)]['accuracy'].tolist()
    assert len(x) == 1
    return x[0]

for metric_name in ['bleu', 'comet','bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'chrf']:
    metric_dataframe = pd.DataFrame.from_records([
        {
            'author': author,
            **{
                model: get_score(model=model, author=author, metric_name=metric_name)
                for model in models
            }
        }
        for author in authors
    ])
    metric_dataframe.to_excel(f'/home/raed_mughaus/Files/Authorship Results/author X model/{metric_name}.xlsx', index=False)


sta_dataframe = pd.concat([
    pd.read_excel(path)
    for path in [
        '/home/raed_mughaus/Files/Authorship Results/Author Classification/STA of allam/sta_per_author.xlsx',
        '/home/raed_mughaus/Files/Authorship Results/Author Classification/STA of Jais and Llama/sta_per_author.xlsx',
        '/home/raed_mughaus/Files/Authorship Results/Author Classification/STA of AceGPT-v2-8B-Chat/sta_per_author.xlsx',
    ]
])

authors = sorted(dataframe['author'].unique().tolist())
models = sorted(dataframe['model'].unique().tolist())

sta_dataframe = pd.DataFrame.from_records([
        {
            'author': author,
            **{
                model: get_accuracy(model=model, author=author)
                for model in models
            }
        }
        for author in authors
    ])
sta_dataframe.to_excel(f'/home/raed_mughaus/Files/Authorship Results/author X model/STA.xlsx', index=False)


# def get_sta(model, author, metric_name):
#     x = dataframe[(dataframe['model'] == model) & (dataframe['author'] == author)][metric_name][0]
#     assert len(x) == 0
#     return x[0]
#
#
# sta_dataframe = pd.DataFrame