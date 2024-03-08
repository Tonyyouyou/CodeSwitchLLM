import os
import sys
import torch
import transformers
from typing import List
from torch.utils.data import random_split
from datasets import load_dataset, concatenate_datasets
from prompter import Prompter
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_int8_training
)

num_epochs = 10

batch_size = 128
micro_batch_size = 8

lora_r = 4
lora_alpha = 4
lora_dropout = 0.05

learning_rate = 2e-4

cutoff_len = 512
val_set_size = 100

base_model =  "/g/data/wa66/Xiangyu/Data/llama2_7B_chinese"
data_path = "/g/data/wa66/Xiangyu/Data/code_switch/train_asru.json"
dev_path = '/g/data/wa66/Xiangyu/Data/code_switch/test_asru.json'
cache_dir='/g/data/wa66/Xiangyu/huggingface_cache'

prompt_template_name = '/home/561/xz4320/Code_Switch/asru/template'

output_dir = "/g/data/wa66/Xiangyu/Result/code_switch_result/result/asruhint_dev"
logging_dir = '/g/data/wa66/Xiangyu/Result/code_switch_result/logs/asruhint_dev'

resume_from_checkpoint = None

gradient_accumulation_steps = batch_size // micro_batch_size
prompter = Prompter(prompt_template_name)
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size


data = load_dataset("json", data_files=data_path,cache_dir=cache_dir)

dev = load_dataset("json", data_files=dev_path,cache_dir=cache_dir)

model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map, 
                                            #  load_in_8bit=True, 
                                            #  torch_dtype=torch.float16
                                             )
config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","v_proj"]
    )
# model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)


tokenizer = AutoTokenizer.from_pretrained(base_model,use_fast=False)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        input=data_point["input"],
        label=data_point["output"],
        hint=data_point['hint']
    )
    tokenized_full_prompt = tokenize(full_prompt)

    return tokenized_full_prompt

if resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = (
            False  # So the trainer won't try loading its state
        )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        print(torch.cuda.get_device_name(0))
        adapters_weights = torch.load(checkpoint_name, map_location='cuda:2')
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")

# if val_set_size > 0:
#     train_val = data["train"].train_test_split(
#         test_size=val_set_size, shuffle=True, seed=42
#     )
#     train_data = (
#         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
#     )
#     val_data = (
#         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
#     )

train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
dev_data = dev["train"].shuffle().map(generate_and_tokenize_prompt)

if not ddp and torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

model.print_trainable_parameters()  # Be more transparent about the % of trainable params

training_args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            fp16=True,
            save_total_limit=3,
            logging_dir = logging_dir,
            logging_steps=100,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=100,
            output_dir=output_dir)
data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

trainer = transformers.Trainer(
    model = model,
    train_dataset=train_data,
    eval_dataset = dev_data,
    args = training_args,
    data_collator=data_collator,
)

model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.save_state()

model.save_pretrained(output_dir)