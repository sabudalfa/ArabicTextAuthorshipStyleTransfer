from os.path import *
from simpleio import *
import pandas as pd


paths = [
    #'/home/raed_mughaus/PycharmProjects/jrcai_text_generation/output/formality-transfer/1719732038/fine-tuned-using-single-dialect/',
    #'/home/raed_mughaus/PycharmProjects/jrcai_text_generation/output/formality-transfer/1719732038/fine-tuned-using-five-dialects/',
    #'/home/raed_mughaus/PycharmProjects/jrcai_text_generation/output/formality-transfer/1719815562/five-shots',
    '/home/raed_mughaus/PycharmProjects/jrcai_text_generation/output/formality-transfer/1719815562/zero-shot',
]

dataframes = []
for jobs_path in paths:
    for job_path in list_dir(jobs_path, return_paths=True):
        if isdir(job_path):
            dataframes.append(pd.DataFrame({
                **read_from_json_file(f'{job_path}/text_generation_job.json'),
                'source': read_from_json_file(f'{job_path}/sources.json'),
                'actual_target': read_from_json_file(f'{job_path}/actual_targets.json'),
                'predicted_target': read_from_json_file(f'{job_path}/predicted_targets.json'),
            }))

dataframe = pd.concat(dataframes)
print(dataframe.columns)
# for source in dataframe['actual_target'].unique():
#     print(source)

selected_targets = [
    'هل يمكن أن أحجز سيارة للإيجار من هنا ؟',
    #'هل هناك جولة بالحافلة تجوب المدينة ؟',
]

for actual_target in selected_targets:
    filtered_dataframe = dataframe[
        (dataframe['actual_target'] == actual_target) &
        (dataframe['dialect'] == 'Doha')
    ]
    print(filtered_dataframe['source'].unique())
    for model, predicted_target in zip(filtered_dataframe['model'], filtered_dataframe['predicted_target']):
        print(f'zero-shot,{model},{predicted_target}')
