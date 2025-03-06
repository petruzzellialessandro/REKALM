import os
import yaml
import json
import torch
import numpy as np
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from transformers.utils import logging
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm
import pickle

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
from utils.utils import POSSIBLE_DTYPES
from dataclass.EntityNames import set_dataset

yaml_path = "config_extraction.yaml"

def main():
    yaml_data = get_config_yaml()
    seed = yaml_data.get("seed", 22)
    set_all_seeds(seed)
    out_file_name = yaml_data.get("out_file_name", 'results')
    model_name = yaml_data.get("model", '')
    domain = yaml_data.get("domain", None)
    tokenizer_name = yaml_data.get("tokenizer", '')
    output_dir = yaml_data.get("output_dir", '')
    dtype = POSSIBLE_DTYPES[yaml_data.get("dtype", "auto")]
    attn_implementation = yaml_data.get("attn_implementation", "eager")
    batch_size = yaml_data.get("batch_size", 128)
    batch_start = yaml_data.get("batch_start", 0)
    dataset_config = yaml_data.get("dataset", {})
    dataset = set_dataset(dataset_config = dataset_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        attn_implementation=attn_implementation,
        torch_dtype=dtype
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True,  use_fast=False, use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model Loaded")
    inputs_ids = np.arange(len(dataset))
    accelerator = Accelerator()
    model, dataset = accelerator.prepare(model, dataset)
    with accelerator.split_between_processes(inputs_ids) as inp:
        results=dict(embedding=[], item_id=[], item_name = [], input_raw=[])
        inputs_text = np.array([x[1] for x in dataset])
        inputs_item_ids = np.array([x[0] for x in dataset])
        proc_inputs_ids = inp
        proc_inputs_items_ids = inputs_item_ids[inp]
        proc_inputs_values = inputs_text[inp]
        with torch.no_grad():
            for i in tqdm(range(0, len(proc_inputs_values), batch_size)):
                logger.info(f"Batching {i}-{i+batch_size}")
                if batch_start > i:
                    if batch_start > i+batch_size:
                        continue
                    else:
                        i = batch_start
                curr_inp_keys = proc_inputs_ids[i:i+batch_size]
                curr_inp_values = proc_inputs_values[i:i+batch_size]
                curr_inp_user_id = proc_inputs_items_ids[i:i+batch_size]
                input_tok = [f'{tokenizer.bos_token} ' + x  for x in curr_inp_values]
                inputs = tokenizer(input_tok, return_tensors="pt", padding=True)
                output = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden_state = output.hidden_states[-1]
                idx_of_the_last_non_padding_token = inputs.attention_mask.bool().sum(1)-1
                sentence_embeddings = last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx_of_the_last_non_padding_token]
                results['input_raw'].extend(input_tok)
                results['embedding'].extend(sentence_embeddings)
                results['item_id'].extend(curr_inp_user_id)
                results['item_name'].extend(curr_inp_values)
        results=[ results ]
    results_gathered=gather_object(results)[0]
    results = [{"item_name": item_name, "item_id": item_id, "embedding": emb, "input_raw": input_raw} for item_name, item_id, emb, input_raw \
     in zip(results_gathered['item_name'], results_gathered['item_id'], results_gathered['embedding'], results_gathered['input_raw'])]
    
    import pandas as pd
    pd.to_pickle(results, os.path.join(output_dir, out_file_name))

def set_all_seeds(seed):
    set_seed(seed=seed)

def get_config_yaml():
    with open(yaml_path, "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data

if __name__ == "__main__":
    main()