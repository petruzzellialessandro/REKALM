import os
import yaml
import json
import torch
import numpy as np
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, pipeline
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
from utils.utils import POSSIBLE_DTYPES
from dataclass.CustomDataset import prepare_dataset_test

yaml_path = "config_inference.yaml"

def main():
    yaml_data = get_config_yaml()
    seed = yaml_data.get("seed", 22)
    set_all_seeds(seed)
    out_file_name = yaml_data.get("out_file_name", 'results')
    model_name = yaml_data.get("model", '')
    domain = yaml_data.get("domain", "")
    tokenizer_name = yaml_data.get("tokenizer", '')
    output_dir = yaml_data.get("output_dir", '')
    import shutil
    shutil.copy2(yaml_path, output_dir)
    dtype = POSSIBLE_DTYPES[yaml_data.get("dtype", "auto")]
    attn_implementation = yaml_data.get("attn_implementation", "eager")
    batch_size = yaml_data.get("batch_size", 128)
    batch_start = yaml_data.get("batch_start", 0)
    generation_parameters = yaml_data.get("generation_parameters", {})
    dataset_config = yaml_data.get("dataset", {})
    num_gpu = 4
    dataset = prepare_dataset_test(data_config = dataset_config)['train']
    inputs_ids = np.arange(len(dataset['text']))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=dtype
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False, use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'
    logger.info("Model Loaded")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto", return_full_text=False, **generation_parameters)
    batch_indexs = [(0+i, min(batch_size*num_gpu+i, len(dataset['text']))) for i in range(0, len(dataset['text']), batch_size*num_gpu)]
    logger.info("Generation params")
    logger.info(json.dumps(generation_parameters))
    with open(os.path.join(output_dir,f"{out_file_name}.jsonl"), "a") as final:
        for min_, max_ in tqdm.tqdm(batch_indexs):
            logger.info(f"Batching {min_}-{max_}")
            if batch_start > min_:
                if batch_start > max_:
                    continue
                else:
                    min_ = batch_start
            text = dataset['text'][min_:max_]
            user_id = inputs_ids[min_:max_]
            for outs, user_id, prompt in zip(pipe(text, batch_size=batch_size), user_id, text):
                x = {"prompt": prompt, "response": outs[0]['generated_text'], "user_id": str(user_id)} 
                json.dump(x, final)
                final.write('\n')
    
def set_all_seeds(seed):
    set_seed(seed=seed)

def get_config_yaml():
    with open(yaml_path, "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data

if __name__ == "__main__":
    main()