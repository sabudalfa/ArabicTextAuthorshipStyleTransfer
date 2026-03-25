from os.path import isdir

import pandas as pd
from simpleio import *


dict_1 = read_from_json_file('../output/experiments/30 - author-classification/output/Meta-Llama-3-8B/evaluation.json')
doct_2 = read_from_json_file('../output/experiments/30 - author-classification/output/jais-family-13b-chat/evaluation.json')

pd.DataFrame([
    {
        'model_name': model_name,
        **read_from_json_file(f'../output/experiments/30 - author-classification/output/{model_name}/evaluation.json'),
    }
    for model_name in list_dir('../output/experiments/30 - author-classification/output/')
]).to_excel(f'../output/experiments/30 - author-classification/summary.xlsx')
