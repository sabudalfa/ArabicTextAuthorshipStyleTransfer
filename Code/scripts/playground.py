import pandas as pd
from llm import BleuMetric
from statistics import mean

dataframe = pd.read_excel('../output/experiments/28 - jais and llama - by book/experiment.xlsx')
dataframe = dataframe[:1000]
ground_truth = dataframe['ground_truth_output'].tolist()
predicted = dataframe['predicted_output'].tolist()
metric = BleuMetric()

for batch_size in [1, 10, 100, 500, 1000]:
    ground_truth_batches = [
        ground_truth[i:i+batch_size]
        for i in range(0, len(ground_truth), batch_size)
    ]
    predicted_batches = [
        predicted[i:i+batch_size]
        for i in range(0, len(predicted), batch_size)
    ]
    print(mean([
        metric(ground_truth=ground_truth_batch, predictions=predicted_batch, sources=[])["bleu"]
        for ground_truth_batch, predicted_batch in zip(ground_truth_batches, predicted_batches)
    ]))

dataframe = pd.read_excel('../output/experiments/29 - jais and llama - by book/summary.xlsx')
dataframe = dataframe.drop(columns=['book_name', 'Unnamed: 0'])
dataframe.groupby(by=['author', 'model']).mean().reset_index().to_excel('./sheet.xlsx')

jais_dataframe = pd.read_excel('../output/experiments/27 - jais-family-13b-chat/summary.xlsx')
llama_dataframe = pd.read_excel('../output/experiments/26 - llama-3.1-8b-instruct/summary.xlsx')

authors = set(jais_dataframe['author'].tolist() + llama_dataframe['author'].tolist())
jais_models = set(jais_dataframe['model'].tolist())
llama_models = set(llama_dataframe['model'].tolist())
for author in authors:
    for model in jais_models:
        filtered_dataframe = dataframe[(dataframe['author'] == author) & (dataframe['model'] == model)]
        filtered_jais_dataframe = jais_dataframe[(jais_dataframe['author'] == author) & (jais_dataframe['model'] == model)]
        assert len(filtered_jais_dataframe) == 1
        if abs(filtered_dataframe['bleu'].mean() - filtered_jais_dataframe.iloc[0].bleu) > 1:
            print(author, model)
    for model in llama_models:
        filtered_dataframe = dataframe[(dataframe['author'] == author) & (dataframe['model'] == model)]
        filtered_llama_dataframe = llama_dataframe[(llama_dataframe['author'] == author) & (llama_dataframe['model'] == model)]
        assert len(filtered_llama_dataframe) == 1
        if abs(filtered_dataframe['bleu'].mean() - filtered_llama_dataframe.iloc[0].bleu) > 1:
            print(author, model)
