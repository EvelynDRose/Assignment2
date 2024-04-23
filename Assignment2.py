# Imports
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer,  BitsAndBytesConfig, AutoTokenizer
import evaluate
import torch
from peft import LoraConfig
from trl import SFTTrainer
from tabulate import tabulate
from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt 
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = ''

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# dataset
dataset = load_dataset('json', data_files='alpaca_data.json', split='train')

print("DATASET: ", dataset)

# models and tokeizers
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model8 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map="auto", num_hidden_layers=8)
model16 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map="auto", num_hidden_layers=16)
model24 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map="auto", num_hidden_layers=24)
model32 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", add_eos_token=True, padding_side='left')

print(model32)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model8.resize_token_embeddings(len(tokenizer))
    model16.resize_token_embeddings(len(tokenizer))
    model24.resize_token_embeddings(len(tokenizer))
    model32.resize_token_embeddings(len(tokenizer))
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# mapping
def formatting_func(example):
    text = ""
    if example['input'] != "":
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {example['instruction']}\n ### Input: {example['input']}\n### Answer: {example['output']}</s>"
    else:
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {example['instruction']}\n\n### Answer: {example['output']}</s>"
    return text
def generate_and_tokenize_prompt(prompt):
    result = tokenizer(formatting_func(prompt), truncation=True, max_length=512, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

# Evaluate
metric_b = evaluate.load("bleu")
metric_r = evaluate.load('rouge')
metric_be = evaluate.load("bertscore")
b, r, be= [],[],[]
def compute_metrics(eval_pred, model):
    correct = 0
    preds, decoded_preds = [], []
    for i in range(len(eval_pred)):
        print("TEXT: ", i)
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {eval_pred['instruction'][i]}\n### Input: {eval_pred['input'][i]}\n### Answer:"
        text2 = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {eval_pred['instruction'][i]}\n### Input: {eval_pred['input'][i]}\n### Answer:{eval_pred['output'][i]}"
        preds.append(text2)
        model_input = tokenizer(text, return_tensors="pt").to("cuda")
        generation = model.generate(model_input.input_ids, max_new_tokens=len(model_input.input_ids[0]), return_dict_in_generate=True, output_scores=True, repetition_penalty=1.2)
        response = tokenizer.decode(generation[0][0], skip_special_tokens=True, eos_token_id=50256)
        gpt = ask_gpt({eval_pred['output'][i]}, response)
        if(gpt[:2] == "Yes"):
            correct = correct + 1

        decoded_preds.append(response)
        print(response)
        print()
    correct = correct/10
    print("Correct = ", correct)

    b.append(metric_b.compute(predictions=preds, references=decoded_preds))
    r.append(metric_r.compute(predictions=preds, references=decoded_preds))
    be.append(metric_be.compute(predictions=preds, references=decoded_preds, lang="en"))

def find_probs(eval_pred, model):
    for i in range(len(eval_pred)):
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction: {eval_pred['instruction'][i]}\n### Input: {eval_pred['input'][i]}\n### Answer:"
        model_input = tokenizer(text, return_tensors="pt").to("cuda")
        generation = model.generate(model_input.input_ids, max_new_tokens=len(model_input.input_ids[0]), return_dict_in_generate=True, output_scores=True, repetition_penalty=1.2)
        transition_scores = model.compute_transition_scores(generation.sequences, generation.scores, normalize_logits=True)
        word, percent = [],[]
        input_length = model_input.input_ids.shape[1]
        generated_tokens = generation.sequences[:, input_length:]
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            decode = tokenizer.decode(tok, skip_special_tokens=True, eos_token_id=50256)
            word.append(decode)
            percent.append( np.exp(score.cpu().numpy()))
        responses.append(word)
        probabilities.append(percent)

def ask_gpt(knowledge, answer):
    prompt = f""" Yes or No. Is the following answer correct given this context:  ```{knowledge}```"""
    response = get_completion(prompt + answer)
    print("GPT responce: " + response)
    return response

# Evaluations set for metrics
eval_set = dataset.shuffle(seed=42).select(range(10))
print(eval_set['instruction'])

# tokenize dataset
tokenized_dataset = dataset.map(generate_and_tokenize_prompt)

print(tokenized_dataset)

table = []
layers = [32]
model = [model32]
data = []
for i in range(len(layers)):
    responses, probabilities = [],[]
    compute_metrics(eval_set, model[i])
    find_probs(eval_set, model[i])
    data.append([responses, probabilities])
    # table
    head =  ["Layer",      "BLEU",      "Rogue-L",              "BERTScore"]
    row =   ["Layer " + str(layers[i]),b[i]['bleu'], r[i]['rougeL'], mean(be[i]["precision"])]

    table.append(row)

print(tabulate(table, headers=head, tablefmt="grid"))

for i in range(10):
    print(data[0][0][i][:10], data[0][1][i][:10])
    print(data[1][0][i][:10], data[1][1][i][:10])
    print(data[2][0][i][:10], data[2][1][i][:10])
    print(data[3][0][i][:10], data[3][1][i][:10])
    fig, axs = plt.subplots(4, 1, figsize = (10, 10))
    # data[model][responce/probs][sentences][first 10 words]
    axs[0].bar(data[0][0][i][:10], data[0][1][i][:10], label = data[0][0][i][:10])
    axs[0].set_xlabel("Probabilites")
    axs[0].set_ylabel("Layer 8")
    axs[1].bar(data[1][0][i][:10], data[1][1][i][:10], label = data[1][0][i][:10])
    axs[1].set_xlabel("Probabilites")
    axs[1].set_ylabel("Layer 16")
    axs[2].bar(data[2][0][i][:10], data[2][1][i][:10], label = data[2][0][i][:10])
    axs[2].set_xlabel("Probabilites")
    axs[2].set_ylabel("Layer 24")
    axs[3].bar(data[3][0][i][:10], data[3][1][i][:10], label = data[3][0][i][:10])
    axs[3].set_xlabel("Probabilites")
    axs[3].set_ylabel("Layer 32")
    plt.show()

######################
