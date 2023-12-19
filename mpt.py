from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
from transformers import  BitsAndBytesConfig
import torch

from utils import normalize, extract_model_response, remove_before_first_inst
import os

from utils import clean_sentence, clean_punctuation
from tqdm import tqdm

def extract_response(content):
    # Split the content by lines
    lines = content.split('\n')

    # Flag to indicate if the response section has started
    in_response = False

    # To store the extracted response
    extracted_response = ''

    for line in lines:
        # Check if the current line indicates the start of the response section, ignoring leading whitespace
        if '### Response:' in line.strip():
            in_response = True
            continue

        # If we are in the response section and the line is not empty, extract the content
        if in_response:
            if line.strip():  # Check if the line is not empty
                extracted_response = line.strip()
                break  # Stop after finding the first non-empty line

    return extracted_response


def generate(prompt, model, tokenizer):

    llama2_template = f"""
    ### Instruction: You should correct the grammar errors in the sentence below. You can change the sentence as little as possible.

    ### Input:
    {prompt}

    ### Response:
    """
    inputs = tokenizer(llama2_template, return_tensors="pt").to("cuda")
    with torch.no_grad():
        #output = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.1, top_k=5, num_return_sequences=1) this is for peft
        output = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=0.1, top_k=1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = output[0].to("cpu")
    printed = tokenizer.decode(output, skip_special_tokens=True)
    printed = extract_response(printed)
    return printed

model_path = "/workspace/mpt-7b"   #path to the model, in hf format


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,quantization_config=BitsAndBytesConfig(load_in_4bit=True), load_in_4bit=True, device_map="cuda:0")

model.eval()
lines = []
fp = "mpt"
with open('test_data/standard/conll14st-test.tok.src', 'r') as f:
    for line in f:
        lines.append(line.strip())

target = []
for line in tqdm(lines, desc="Processing lines"):
    text = (generate(line, model, tokenizer))
    if text is not None:
        text = normalize(text)
    else:
        text = line
    print(text)
    target.append(text)

with open(f'test_data/0-shot/{fp}output_norm.txt', 'w') as f:
    for line in target:
        f.write(line + '\n')