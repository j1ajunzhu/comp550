from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
from transformers import  BitsAndBytesConfig
import torch

from utils import normalize, extract_model_response, remove_before_first_inst
import os

from utils import clean_sentence, clean_punctuation
from tqdm import tqdm
import argparse


def generate(prompt, model, tokenizer, model_type):
    generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 64,
    "num_return_sequences": 1,}
    prompt = f"<s> GEC: {prompt}\nCorrected:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        #output = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.1, top_k=5, num_return_sequences=1) this is for peft
        #output = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=0.1, top_k=1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        output = model.generate(**inputs, **generation_kwargs)
    output = output[0].to("cpu")
    printed = tokenizer.decode(output, skip_special_tokens=True)
    print(printed)
    if model_type == "llama":
        printed = (printed.split('\n')[1])
        from utils import retain_content_before_last_non_english_punctuation
        printed = retain_content_before_last_non_english_punctuation(printed)
        return printed.replace("Corrected: ", "")
    
    elif model_type == "mpt" or model_type == "falcon":
        printed = printed.split('\n')[1]
        printed = clean_sentence(printed)
        printed = clean_punctuation(printed)
        printed = printed.replace("</s>", " ")
        printed = printed.replace("</s", " ")
        printed = printed.replace("</", " ")
        return printed
    


parser = argparse.ArgumentParser()

parser.add_argument('--model_path', help='Your name')
parser.add_argument('--model_type', help='Your name')
args = parser.parse_args()

model_path = args.model_path   #path to the model, in hf format


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,quantization_config=BitsAndBytesConfig(load_in_4bit=True), load_in_4bit=True, device_map="cuda:0")
model.eval()
lines = []
fp = os.path.basename(model_path).split('.')[0]

with open('test_data/standard/conll14st-test.tok.src', 'r') as f:
    for line in f:
        lines.append(line.strip())

target = []
for line in tqdm(lines, desc="Processing lines"):
    text = normalize(generate(line, model, tokenizer, args.model_type))
    print(text)
    target.append(text)


with open(f'test_data/saveresult/sftresult/{fp}.txt', 'w') as f:
    for line in target:
        f.write(line + '\n')