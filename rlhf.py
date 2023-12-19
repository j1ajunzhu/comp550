# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_xpu_available

from peft import PeftModelForCausalLM, PeftConfig
from transformers import  BitsAndBytesConfig, AutoModelForCausalLM
import datasets
import numpy as np
tqdm.pandas()


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="lvwerra/gpt2-imdb",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5,
            log_with=None,
            mini_batch_size=4,
            batch_size=4,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    use_seq2seq: bool = False
    """whether to use seq2seq models"""
    use_peft: bool = True
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules = ["q_proj", "v_proj"],
            lora_dropout=0.05,
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})


args = tyro.cli(ScriptArguments)


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = datasets.load_from_disk("rlhfdataset")
    ds = ds.shuffle(seed=15580)
    ds = ds.select(range(5000))

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["text"])
        source = sample["source"]
        target = sample["target"]
        sample["query"] = f"source: {source} \n target: {target}"
        return sample
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.ppo_config.model_name,
    device_map={"": "xpu:0"} if is_xpu_available() else {"": 0},
    peft_config=lora_config,
    quantization_config=nf4_config,
    use_safetensors=True,
    load_in_4bit=True,
)

"""
ref_model = AutoModelForCausalLM.from_pretrained(
    args.ppo_config.model_name,
    device_map={"": "xpu:0"} if is_xpu_available() else {"": 0},
    quantization_config=nf4_config,
    use_safetensors=True,
    load_in_4bit=True,
)
"""
tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token


# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model = None, tokenizer=tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = args.ppo_config.reward_model.split(":")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

import errant
annotator = errant.load('en')


def errant(annotator, orig, cor):
    orig = annotator.parse(orig)
    cor = annotator.parse(cor)
    edits = annotator.annotate(orig, cor)
    return len(edits)

def retain_content_before_last_non_english_punctuation(text):
    # Reverse iterate over the text from the end
    for i in range(len(text) - 1, 0, -1):
        # Check if the current character is a space and the previous one is an English letter
        if text[i] == ' ' and text[i-1].isalpha():
            # Return the text up to the space
            return text[:i+2]
    # Return the original text if no such space is found
    return text


def sigmoid_with_controlled_threshold(value, threshold=0.8, control_point=5):
    """
    Apply a sigmoid function with a controlled threshold, ensuring that the sigmoid
    value of the control point is exactly the specified threshold.

    Args:
    - value (float): The value to apply the sigmoid to.
    - threshold (float): The desired sigmoid value at the control point.
    - control_point (float): The value at which the sigmoid should equal the threshold.

    Returns:
    - float: The sigmoid-transformed value.
    """
    # Calculating the required scale factor based on the control point and threshold
    scale = -np.log((1 / threshold) - 1) / control_point

    # Applying the scaled sigmoid function
    return 1 / (1 + np.exp(-scale * value))

def calculate_rewards(source, target):
    normalized_response = []
    rewards_li = []
    for input, output in zip(source, target):
        standard = input.split('\n')[1]
        standard = standard.replace(" target: ", "")
        corrupt = input.split('\n')[0]
        corrupt = corrupt.replace("source: ", "")
        source_correctness = errant(annotator, standard, corrupt)
        if "\n" in output:
            output = output.split("\n")[0]
        output = retain_content_before_last_non_english_punctuation(output)

        output = output.replace("Corrected", "")
        actual_correctness = errant(annotator, standard, output)

        if source_correctness == 0 and actual_correctness != 0:
            rewards_li.append( 1 - sigmoid_with_controlled_threshold(actual_correctness, threshold=0.666, control_point=3))
        elif source_correctness == 0 and actual_correctness == 0:
            rewards_li.append(1)
        else:
            rewards_li.append(1-(actual_correctness/source_correctness))

        normalized_response.append(output)
    return rewards_li, normalized_response

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    #print(query_tensors)
    # Get response from gpt2
    response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    #batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)


    rewards,  batch["normalized_response"] = calculate_rewards(batch["query"], batch["response"])



    rewards = [torch.tensor(output) for output in rewards]

    #ref_rewards = calculate_rewards(batch["query"], batch["ref_response"])
    #ref_rewards = [torch.tensor(output) for output in ref_rewards]
    #batch["ref_rewards"] = ref_rewards
    
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "normalized_response"])

# Save the model
lines = []
with open('test_data/standard/conll14st-test.tok.src', 'r') as f:
    for line in f:
        lines.append(line.strip())

target = []
ppo_trainer.model.eval()
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 64,
}
for line in tqdm(lines, desc="Processing lines"):
    prompt = f"<s> GEC: {line}\nCorrected:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")["input_ids"][0]
    text = ppo_trainer.generate(inputs, **generation_kwargs)
    text = tokenizer.batch_decode(text, skip_special_tokens=True)[0]
    print(text)
    if "\n" in text and text is not None:
        text = text.split("\n")[1]
        text = retain_content_before_last_non_english_punctuation(text)
    elif text is not None:
        text = retain_content_before_last_non_english_punctuation(text)
    elif text is None:
        text = line
    text = text.replace("Corrected: ", "")
    print(f"this is correct: {text}")
    target.append(text)

with open(f'pporesult.txt', 'w') as f:
    for line in target:
        f.write(line + '\n')