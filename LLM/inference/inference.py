import os
import sys
import editdistance
from prompter import Prompter
import json
import torch
import re
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    PeftModel,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_int8_training,
    PeftType,
    PromptEncoderConfig,
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

device_map = "auto"
load_8bit = True

def calculate_wer(pre, ref):
    wer_score = editdistance.eval(pre, ref) / len(ref)
    return wer_score

base_model =  "/g/data/wa66/Xiangyu/Data/llama2_7B_ch"
data_path = "/g/data/wa66/Xiangyu/Data/code_switch/output_test_new.json"
lora_weight = '/g/data/wa66/Xiangyu/code_switch_result/result/hint/checkpoint-1500'
prompt_template_name = '/g/data/wa66/Xiangyu/Data/code_switch/with_correct_hint'

cache_dir='/g/data/wa66/Xiangyu/huggingface_cache'

prompter = Prompter(prompt_template_name)
tokenizer = AutoTokenizer.from_pretrained(base_model,use_fast=False)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map,load_in_8bit=load_8bit,torch_dtype=torch.float16)

model = PeftModel.from_pretrained(
    model,
    lora_weight,
    torch_dtype=torch.float16,
    device_map=device_map,
)

testset = 'code_swicth'
os.system(f'mkdir -p calibrate/{testset}_final')
f_hypo = open(f'calibrate/{testset}_final/hypo', 'w')
f_gt = open(f'calibrate/{testset}_final/gt', 'w')

model.config.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def alpaca_infer(input1, input2=None):

    prompt = prompter.generate_prompt(input=input1)
    # print('prompt', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    # print('inputs', inputs)

    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        stream_output=False,
        prompter=prompter,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    return prompter.get_response(output)



with open(data_path, 'r') as f:
    test_data = json.load(f)

before = 0
after = 0

ignore = 0

for index in range(len(test_data)):
    print('start decoding')
    best_hypo = test_data[index]['input'][0]
    ground_truth = test_data[index]['output']
    prediction = alpaca_infer(input1=test_data[index]['input'])
    prediction = re.sub('</s>', '', prediction).lower()
    prediction = re.sub(r'[^\w\s]|[\d]', '', prediction)
    best_hypo = re.sub(r'[^\w\s]|[\d]', '', best_hypo)
    ground_truth = re.sub(r'[^\w\s]|[\d]', '', ground_truth)

    try:
        wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
        wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
    except Exception:
        ignore += 1
        continue
    before = before + wer_best_hypo
    after = after + wer_prediction
    if wer_best_hypo != wer_prediction:
        print('before:::', best_hypo)
        print('after :::', prediction)
        print('answer:::', ground_truth)
        print('before score', wer_best_hypo)
        print('after score', wer_prediction)


    f_hypo.write(f'{prediction} (utt_{index})\n')
    f_gt.write(f'{ground_truth} (utt_{index})\n')

print('before',before / (len(test_data) - ignore))
print('after',after / (len(test_data) - ignore))

f_hypo.close()
f_gt.close()