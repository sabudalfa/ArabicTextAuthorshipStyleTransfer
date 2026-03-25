from simpleio import *

EXPERIMENT_OUTPUT_DIR = f'/home/raed_mughaus/PycharmProjects/jrcai_text_generation/output/formality-transfer-small-models/1/lr-optimization/MadarV6/Doha/'

for folder in list_dir(EXPERIMENT_OUTPUT_DIR, return_paths=True):
    learning_rate_loss_tuples = read_from_json_file(
        file_path=f'{folder}/learning_rate_optimization.json',
    )
    lr = min(
        learning_rate_loss_tuples,
        key=lambda record: record[1],
    )[0]
    print(folder.split('/')[-1], lr)
