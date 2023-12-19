from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
from transformers import  BitsAndBytesConfig
import torch

from utils import normalize, extract_model_response, remove_before_first_inst
import os

from tqdm import tqdm

    
def generate(prompt, model, tokenizer):
    llama2_template = f"""
    ### Instruction: You should correct the grammar errors in the sentence below. You can change the sentence as little as possible.

    ### Input:
    {input}

    ### Response:
    """
    inputs = tokenizer(llama2_template, return_tensors="pt").to("cuda")
    with torch.no_grad():
        #output = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.1, top_k=5, num_return_sequences=1) this is for peft
        output = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.9, temperature=0.1, top_k=5, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = output[0].to("cpu")
    printed = tokenizer.decode(output, skip_special_tokens=True)
    print(printed)
    printed = printed.split('\n')[3]
    return printed

model_path = "/workspace/llama1-7b-hf"   #path to the model, in hf format


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,quantization_config=BitsAndBytesConfig(load_in_4bit=True), load_in_4bit=True)
model.eval()
"""
peft_model_id = "/workspace/peft_try" 
model = PeftModelForCausalLM.from_pretrained(model, peft_model_id, device="cuda", load_in_4bit=True, torch_dtype=torch.bfloat16,
                                             llm_int8_skip_modules = None,
                                             llm_int8_enable_fp32_cpu_offload = False,
                                             llm_int8_has_fp16_weight = False,
                                             bnb_4bit_quant_type = "nf4",
                                             bnb_4bit_use_double_quant = True,
                                             bnb_4bit_compute_dtype = torch.bfloat16)
"""
lines = "llama1v2"
fp = os.path.basename(model_path).split('.')[0]
with open('test_data/standard/conll14st-test.tok.src', 'r') as f:
    for line in f:
        lines.append(line.strip())

target = []
for line in tqdm(lines, desc="Processing lines"):
        text = generate(line, model, tokenizer)
        if text is None:
             target.append(line)
             #print(f"not founed{line}")
        else:
            target.append(normalize(text))
            #print(normalize(text))

with open(f'test_data/0-shot/{fp}output_norm.txt', 'w') as f:
    for line in target:
        f.write(line + '\n')