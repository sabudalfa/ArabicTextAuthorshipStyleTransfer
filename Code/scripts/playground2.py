import os.path
from simpleio import *

model_path = '../output/experiments/30 - author-classification/output/jais-family-13b-chat/'
author_paths = [
    author_path
    for author_path in list_dir(model_path, return_paths=True)
    if os.path.isdir(author_path)
]

print(set([
        predicted_author
        for author_path in author_paths
        for predicted_author in read_from_json_file(f'{author_path}/predicted_outputs.json')
    ]))
print(len(set([
    predicted_author
    for author_path in author_paths
    for predicted_author in read_from_json_file(f'{author_path}/predicted_outputs.json')
    ])))

for author_name in set([
        predicted_author
        for author_path in author_paths
        for predicted_author in read_from_json_file(f'{author_path}/predicted_outputs.json')
    ]):
    print(author_name, len([
        predicted_author
        for author_path in author_paths
        for predicted_author in read_from_json_file(f'{author_path}/predicted_outputs.json')
        if predicted_author == author_name
    ]))

write_to_json_file(
    f'{model_path}/ground_truth_authors.json',
    content=[
        ground_truth_output.strip()
        for author_path in author_paths
        for ground_truth_output in read_from_json_file(f'{author_path}/ground_truth_outputs.json')
    ],
)

write_to_json_file(
    f'{model_path}/predicted_authors.json',
    content=[
        predicted_author.strip()
        for author_path in author_paths
        for predicted_author in read_from_json_file(f'{author_path}/predicted_outputs.json')
    ],
)
