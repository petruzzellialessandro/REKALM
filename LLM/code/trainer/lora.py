from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

def set_config(lora_parameters, model, device) -> None:
    r = lora_parameters.get("r", 8)
    lora_alpha = lora_parameters.get("alpha", 32)
    lora_dropout = lora_parameters.get("dropout", 0.1)
    task_type = lora_parameters.get("dropout", "CAUSAL_LM")
    target_modules = lora_parameters.get("target_modules", None)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    if model.is_loaded_in_8bit or model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(peft_config=lora_config, model=model)
    return model, lora_config
