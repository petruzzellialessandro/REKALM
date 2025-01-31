import torch.nn as nn
from datasets import load_dataset, load_from_disk
import json
from tqdm import tqdm
import os

def prepare_dataset_adaptation(data_config):
    data_folder = data_config.get("data_folder", "data")
    file_name = data_config.get("file_name", "")
    file_extension = data_config.get("file_extension", "")
    dataset = load_dataset(file_extension, data_files=os.path.join(data_folder, file_name))
    return dataset.shuffle()


def format_prompt_adaptation(x):
    PROMPT = """The beer %s (id %s) has an alcohol by volume of %s. It's style is %s.\n\nAn user review it as follows: %s"""
    return (PROMPT % (x['beer/name'], x['beer/beerId'], x['beer/ABV'], x['beer/style'], x['review/text']))


def prepare_dataset_task(data_config):
    data_folder = data_config.get("data_folder", "data")
    file_name = data_config.get("file_name", "")
    file_extension = data_config.get("file_extension", "")
    dataset = load_dataset(file_extension, data_files=os.path.join(data_folder, file_name))
    #dataset = load_from_disk(os.path.join(data_folder, file_name))
    return dataset.shuffle()


def prepare_dataset_test(data_config):
    data_folder = data_config.get("data_folder", "data")
    file_name = data_config.get("file_name", "")
    file_extension = data_config.get("file_extension", "")
    dataset = load_dataset(file_extension, data_files=os.path.join(data_folder, file_name))
    return dataset

def format_prompt_task(x):
    #PROMPT = """The beer %s (id %s) has an alcohol by volume of %s. It's style is %s.\n\nAn user review it as follows: %s"""
    return x #(PROMPT % (x['beer/name'], x['beer/beerId'], x['beer/ABV'], x['beer/style'], x['review/text']))