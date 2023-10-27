

import fire

from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import sys
from transformers import GenerationConfig
import json
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import copy
from finetune import (SYS_POSTFIX, 
                      SYS_PREFIX, 
                      INST_POSTFIX, 
                      INST_PREFIX, 
                      OUTPUT_POSTFIX, 
                      OUTPUT_PREFIX,
                      preprocess,
                      generate_response,
                      ask_alpaca,
                      read_json)


def batch_inference(data, model, tokenizer, batch_size = 4):
    tk = tqdm(range(0, len(data), batch_size))
    predictions = []
    for start_idx in tk:
        batch = data[start_idx:start_idx+batch_size]
        preds, _ = ask_alpaca(batch, model, tokenizer)
        predictions += preds
        examples = [p[:50] for p in preds]
        tk.set_postfix(
            examples=examples,
        )
    return predictions

def ValidateFinetunePerformance(model, tokenizer, data, data_name, batch_size = 6, test_limit = -1):
    if isinstance(test_limit, int) and test_limit > -1:
        data = data[:test_limit]

    print("Start validating:", data_name)
    predict_items = []

    for item_id, data_point in enumerate(data):
        dialog = data_point['dialog']
        prompt = ""

        roles = [msg["role"] for msg in dialog]
        messages = [msg["content"] for msg in dialog]

        if roles[0].upper() == "SYSTEM":
            prompt += f"{SYS_PREFIX}{messages[0]}{SYS_POSTFIX}"

        for dialog_pos, (role, msg) in enumerate(zip(roles, messages)):
            if role.upper() == "ASSISTANT":
                predict_items.append({
                    "prompt": prompt,
                    "answer": msg,
                    "item_id": item_id,
                    "dialog_position": dialog_pos
                })
                prompt += f"{msg}{OUTPUT_POSTFIX}"
            elif role.upper() == "USER":
                prompt += f"{INST_PREFIX}{msg}{INST_POSTFIX}{OUTPUT_PREFIX}"

    prompts = [p['prompt'] for p in predict_items]
    results = batch_inference(prompts, model, tokenizer, batch_size = batch_size)

    print("Start prediction")
    for result, predict_item in zip(results, predict_items):
        item_id = predict_item['item_id']
        dialog_position = predict_item['dialog_position']
        predict_dialog = data[item_id].get('predict_dialog')
        if predict_dialog is None:
            data[item_id]['predict_dialog'] = copy.deepcopy(data[item_id]['dialog'])
            predict_dialog = data[item_id]['predict_dialog']

        predict_dialog[dialog_position]['content'] = result

    return data


def validate(
    data_path: str,
    data_name: str,
    lora_model: str,
    batch_size: int,
    OPENAIKEY: str
):
    validate_data = read_json(data_path)
    device_map = "auto"
    config = PeftConfig.from_pretrained(lora_model)
    base_model = config.base_model_name_or_path

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(model, lora_model)
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    ValidateFinetunePerformance(
        model=model,
        tokenizer=tokenizer,
        data=validate_data,
        data_name=data_name,
        batch_size=batch_size
    )
    

if __name__ == "__main__":
    fire.Fire(validate)

   

