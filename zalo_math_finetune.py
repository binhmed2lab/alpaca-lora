import os
import sys
from typing import List
import json
from tqdm import tqdm
import wandb

import fire
import torch
import transformers
from transformers import GenerationConfig
from datasets import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

SYS_PREFIX = "<<SYS>>\n"
SYS_POSTFIX = "\n<</SYS>>\n\n"
INST_PREFIX = "<s>[INST] "
INST_POSTFIX = " "
OUTPUT_PREFIX = "[/INST] "
OUTPUT_POSTFIX = "</s>"

def preprocess(data_point):
    global tokenizer
    cutoff_len = 1280
    dialog = data_point['dialog']

    roles = [msg["role"] for msg in dialog]
    messages = [msg["content"] for msg in dialog]

    assert roles[0].upper() != "ASSISTANT"
    assert roles[-1].upper() == "ASSISTANT"

    input_messages = []
    if roles[0].upper() == "SYSTEM":
        input_messages.append(SYS_PREFIX+messages[0]+SYS_POSTFIX)

    for role, msg in zip(roles, messages):
        if role.upper() == "ASSISTANT":
            input_messages.append(msg + OUTPUT_POSTFIX)
        elif role.upper() == "USER":
            input_messages.append(INST_PREFIX + msg + INST_POSTFIX + OUTPUT_PREFIX)

    tokenized_input = tokenizer(input_messages, add_special_tokens=False)

    input_ids = []
    labels = []

    if roles[0].upper() == "SYSTEM":
        input_ids.extend(tokenized_input.input_ids[0])
        labels.extend([-100]*len(tokenized_input.input_ids[0]))

    for role, msg in zip(roles, tokenized_input.input_ids):

        if role.upper() == "USER":
            labels.extend([-100]*len(msg))
            input_ids.extend(msg)
        
        elif role.upper() == "ASSISTANT":
            labels.extend(msg)
            input_ids.extend(msg)


    input_ids = torch.LongTensor(input_ids)[:cutoff_len]
    labels = torch.LongTensor(labels)[:cutoff_len]

    assert input_ids.shape == labels.shape

    return {
        "input_ids": input_ids,
        "labels": labels
    }

def transformer_to_dialog(math_data):
    dialogs = []

    for d in math_data['data']:
        question = d['question']
        choices = d['choices']
        choices = "\n".join(choices)
        answer = d['answer']
        explanation = d.get("explanation", None)
        dialog = [
            {"role": "system", "content": "Bạn đang trong 1 cuộc thi toán tiểu học. Xin hãy trả lời bằng tiếng Việt."}
        ]
        if explanation:
            dialog += [
            {"role": "user", "content": f"Câu hỏi: {question}\nViết lời giải của bạn."},
            {"role": "assistant", "content": explanation},
            {"role": "user", "content": f"Dựa theo lời giải của bạn, thì lựa chọn nào sau đây là chính xác:{choices}"},
            {"role": "assistant", "content": answer}
            ]
        else:
            dialog += [
            {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn nào sau đây là chính xác:{choices}"},
            {"role": "assistant", "content": answer}
            ]

        dialogs.append(dialog)
        
    return dialogs

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "math_train.json",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: float = 0.3,
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    optim: str = "paged_adamw_32bit",
    # lora hyperparams
    train_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    wandb_api_key: str = None, # Wandb api key
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = torch.cuda.device_count()
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    wandb.login(key=wandb_api_key)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    global tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    data = read_json(data_path)
    data = transformer_to_dialog(math_data=data)
    data = Dataset.from_dict({"dialog": data})
    if val_set_size > 1:
        val_set_size = 0.3
    val_set_size = int(val_set_size * len(data))
    train_val = data.train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(preprocess)
    )
    val_data = (
        train_val["test"].shuffle().map(preprocess)
    )


    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    total_steps = num_epochs * len(train_data) // batch_size
    logging_steps = int(0.1 * total_steps)
    eval_steps = total_steps // num_epochs

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            weight_decay = weight_decay,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            fp16=True,
            max_grad_norm = max_grad_norm,
            logging_steps=logging_steps,
            optim=optim, # adamw_torch
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    

def read_json(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data

def write_json(path, obj):
    if not path.endswith(".json"):
        path += ".json"

    json_object = json.dumps(obj, indent=4, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def generate_response(prompt, model, tokenizer, max_length = 1500):
    encoding = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length = max_length)
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    min_idx = 10000
    for ids in input_ids:
        r = (ids==tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(r) == 0:
          min_idx = 0
          break

        max_idx = max(r)
        min_idx = min(max_idx, min_idx)

    input_ids = input_ids[:,min_idx:]
    attention_mask = attention_mask[:,min_idx:]

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=1,
        do_sample = True,
        num_beams = 1,
        top_k = 50
    )
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )

def format_response(response, tokenizer):
    if response.sequences.size(0) == 1:
        decoded_output = tokenizer.decode(response.sequences[0], skip_special_tokens = True)
        response = [decoded_output.split(OUTPUT_PREFIX)[-1].strip()]
        # put to list to make it compatible
    else:
        decoded_outputs = tokenizer.batch_decode(response.sequences, skip_special_tokens=True)
        response = []
        for o in decoded_outputs:
            response.append(o.split(OUTPUT_PREFIX)[-1].strip())
    return response

def ask_alpaca(prompt, model, tokenizer, max_length = 1500):
    response = generate_response(prompt, model, tokenizer, max_length = max_length)
    response = format_response(response, tokenizer)
    return response


if __name__ == "__main__":
    fire.Fire(train)
