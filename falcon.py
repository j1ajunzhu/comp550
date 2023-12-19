from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
from transformers import  BitsAndBytesConfig
import torch

from utils import normalize, extract_model_response, remove_before_first_inst
import os



    
def generate(prompt, model, tokenizer):
    llama2_template = f"""
    You are a helpful assistant. Your task is to correct my grammar errors with minimum change of text.
    Sentence: {prompt} <s>
    Correct_sentence:
    """
    inputs = tokenizer(llama2_template, return_tensors="pt").to("cuda")
    with torch.no_grad():
        #output = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.1, top_k=5, num_return_sequences=1) this is for peft
        output = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=0.1, top_k=1, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = output[0].to("cpu")
    printed = tokenizer.decode(output, skip_special_tokens=True)
    #print(printed)
    printed = printed.split('\n')[4][7:-4]
    return printed    

model_path = "/workspace/falcon-7b"   #path to the model, in hf format


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,quantization_config=BitsAndBytesConfig(load_in_4bit=True), load_in_4bit=True)
model.eval()

lines = []
fp = os.path.basename(model_path).split('.')[0]
with open('test_data/standard/conll14st-test.tok.src', 'r') as f:
    for line in f:
        lines.append(line.strip())

from tqdm import tqdm
target = []
for line in tqdm(lines):
        text = generate(line, model, tokenizer)
        if text is not None:
             target.append(text)
        else:
             target.append(line)
        print(text)



with open(f'test_data/0-shot/falcon.txt', 'w') as f:
    for line in target:
        f.write(line + '\n')

