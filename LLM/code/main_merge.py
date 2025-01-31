import os
import yaml
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from utils.utils import POSSIBLE_DTYPES

yaml_path = "config_merge.yaml"

def main():
    yaml_data = get_config_yaml()
    seed = yaml_data.get("seed", 22)
    set_all_seeds(seed)
    device = yaml_data.get("device", "auto")  
    domain = yaml_data.get("domain", None)  
    base_model_name_or_path = yaml_data.get("model", '')
    peft_model_path = yaml_data.get("peft_model_path", '')
    output_dir = yaml_data.get("output_dir", None)
    dtype = POSSIBLE_DTYPES[yaml_data.get("dtype", "auto")]


    output_dir = os.path.join(output_dir)
    device_arg = { 'device_map': device }

    print(f"Loading base model: {base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        return_dict=True,
        torch_dtype=dtype,
        **device_arg
    )

    print(f"Loading PEFT: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path, torch_dtype=dtype, **device_arg)

    # to check that the parameters have been merged correctly (if everything is equal to 0 something is wrong)
    lora_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
    for n, p in lora_params.items():
        print(n, p.sum())
        
    print(f"Running merge_and_unload")
    model = model.merge_and_unload(progressbar=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    model.save_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")
    for name in os.listdir(peft_model_path):
        pathname = os.path.join(peft_model_path, name)
        if os.path.isfile(pathname):
            if name.endswith('.yaml'):
                shutil.copy2(pathname, output_dir)
    shutil.copy2(yaml_path, output_dir)
    print(f"Model saved to {output_dir}")


def set_all_seeds(seed):
    set_seed(seed=seed)

def get_config_yaml():
    with open(yaml_path, "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data

if __name__ == "__main__":
    main()