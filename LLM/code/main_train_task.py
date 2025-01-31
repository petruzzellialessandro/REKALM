#general import
import yaml
import os

#specific import
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import torch
from peft import get_peft_model, LoraConfig
from datasets import disable_caching
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

#internal import 
from dataclass.CustomDataset import prepare_dataset_adaptation, format_prompt_adaptation, format_prompt_task, prepare_dataset_task
from trainer.lora import set_config
from utils.utils import POSSIBLE_DTYPES

yaml_path = "config_task.yaml"

def main():
    yaml_data = get_config_yaml()
    seed = yaml_data.get("seed", 22)
    set_all_seeds(seed)
    dataset_config = yaml_data.get("dataset", {})
    device = yaml_data.get("device", "auto")  
    model_name = yaml_data.get("model", '')
    train_type = yaml_data.get("train_type", 'task_adapt')
    tokenizer_name = yaml_data.get("tokenizer", '')
    use_lora = yaml_data.get("use_lora", False)
    load_in_8bit = yaml_data.get("load_in_8bit", False)
    load_in_4bit = yaml_data.get("load_in_4bit", False)
    dtype = POSSIBLE_DTYPES[yaml_data.get("dtype", "auto")]
    attn_implementation = yaml_data.get("attn_implementation", "eager")
    training_args = yaml_data.get("training_args", {})
    output_directory_tr = training_args.get("output_dir", "saved")
    os.makedirs(output_directory_tr, exist_ok=True)
    import shutil
    shutil.copy2(yaml_path, output_directory_tr)
    logger.info(training_args)
    if training_args.get("gradient_checkpointing", False):
            training_args['gradient_checkpointing_kwargs']={"use_reentrant": False}
    lora_parameters = yaml_data.get("lora_parameters", {})
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=None, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, attn_implementation=attn_implementation, use_cache=False, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True,  use_fast=False, use_cache=False)
    logger.info("Model Loaded")
    logger.info(f"Training with LoRA: {use_lora}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    lora_config = None
    if use_lora:
        lora_config = LoraConfig(
                **lora_parameters
            )
        model = get_peft_model(model, lora_config)
    #model = get_peft_model(model, lora_config)
    if train_type == "task_adapt":
        formatting_func = lambda x: format_prompt_task(x)
        dataset = prepare_dataset_task(data_config = dataset_config)['train']
        #logger.info(dataset['text'])
    else:
        formatting_func = lambda x: format_prompt_adaptation(x)
        dataset = prepare_dataset_adaptation(data_config = dataset_config)['train']
    logger.info(dataset[0]['text'])
    trainer_conf = yaml_data.get("SFTtrainer", {})
    args_ = TrainingArguments(
        **training_args
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        #formatting_func=formatting_func,
        dataset_text_field="text",
        dataset_kwargs={
        "add_special_tokens": False
        },
        **trainer_conf,
        args=args_
    )
    logger.info("Start Training")
    if training_args.get("resume_from_checkpoint", False) == False:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_directory_tr)


def get_config_yaml():
    with open(yaml_path, "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data

def set_all_seeds(seed):
    set_seed(seed=seed)

if __name__ == '__main__':
    main()