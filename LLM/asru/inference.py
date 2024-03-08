import os
import sys
import editdistance
from prompter import Prompter
import json
import torch
import re
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

device_map = "auto"
# load_8bit = True

def calculate_wer(pre, ref):
    wer_score = editdistance.eval(pre, ref) / len(ref)
    return wer_score

base_model =  "/home3/hexin/chinese-llama-2-7b"
data_path = "/home3/hexin/data_llm/output_test_new.json"
lora_weight = '/home3/hexin/data_llm/checkpoint-1500'
prompt_template_name = '/home3/hexin/data_llm/with_correct_hint'
use_hint = True


cache_dir='/home3/hexin/huggingface_cache'
# max_memory_mapping = {0: "40GB"}
prompter = Prompter(prompt_template_name)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model, 
    device_map='auto',
    # load_in_8bit=load_8bit,
    # torch_dtype=torch.float16,
    # max_memory=max_memory_mapping,
    )

model = PeftModel.from_pretrained(
    model,
    lora_weight,
    torch_dtype=torch.float16,
    # device_map=device_map,
)

testset = 'code_swicth'
os.system(f'mkdir -p calibrate/{testset}_final')


model.config.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def alpaca_infer(hint, input1):

    prompt = prompter.generate_prompt(hint=hint, input=input1)
    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        stream_output=False,
        prompter=prompter,
        # model=model,
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

f_hypo = open(f'calibrate/{testset}_final/hypo', 'w')
f_gt = open(f'calibrate/{testset}_final/gt', 'w')

print(f'start decoding for {len(test_data)} test samples')
for index in tqdm(range(len(test_data))):
    best_hypo = test_data[index]['input'][0]
    ground_truth = test_data[index]['output']
    if use_hint:
        hint = test_data[index]['hint']
    else:
        hint = None

    prediction = alpaca_infer(hint=hint, input1=test_data[index]['input'])
    # print(prediction)

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