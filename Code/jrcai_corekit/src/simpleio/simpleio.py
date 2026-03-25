import json
import os
from os.path import exists, isdir
from typing import *
from pathlib import Path


def make_parent_dirs(path: str):
    (Path(path)
     .parent
     .mkdir(parents=True, exist_ok=True))


def delete_file(path: str):
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def list_dir(path: str, default_value=None, return_paths=False):
    if not isdir(path):
        return default_value
    if return_paths:
        return [
            f'{path}/{file_name}'
            for file_name in os.listdir(path)
        ]
    else:
        return os.listdir(path)


def read_lines_from_file(file_path: str):
    if not exists(file_path):
        return None
    with open(file_path, mode='r') as file:
        return [
            line.strip()
            for line in file.readlines()
        ]


def read_from_file(file_path: str, default_value=None):
    if not exists(file_path):
        return default_value
    with open(file_path, mode='r') as file:
        return file.read()


def read_from_json_file(file_path: str, default_value=None):
    if not exists(file_path):
        return default_value
    with open(file_path, mode='r') as json_file:
        return json.load(json_file)


def write_to_file(file_path: str, content: Union[str, List[str]], mode='w'):
    make_parent_dirs(file_path)
    with open(file_path, mode=mode) as file:
        if type(content) is str:
            file.write(content)
        else:
            for line in content:
                file.write(f'{line}\n')


def append_to_file(file_path: str, content: Union[str, List[str]]):
    write_to_file(file_path, content, mode='a')


def write_to_json_file(file_path: str, content):
    make_parent_dirs(file_path)
    with open(file_path, mode='w') as json_file:
        return json.dump(content, json_file, ensure_ascii=False)


def append_to_json_list_file(file_path: str, items):
    write_to_json_file(
        file_path=file_path,
        content=read_from_json_file(
            file_path,
            default_value=[]
        ) + items,
    )
