# I cannot send emails or create external downloads, but I can create the files directly here
# Let me create all the necessary files as individual downloadable assets

print("=== CREATING INDIVIDUAL DOWNLOADABLE FILES ===\n")

# 1. Create eval.py content
eval_py_content = '''import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def sweep(score, x):
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    fpr, tpr, auc_val, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))
    low = tpr[np.where(fpr<.05)[0][-1]]
    print('Attack %s AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\\n'%(legend, auc_val, acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc_val
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text)
    return legend, auc_val, acc, low

def fig_fpr_tpr(all_output, output_dir):
    print("output_dir", output_dir)
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])
    
    plt.figure(figsize=(4,3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            legend, auc_val, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write('%s AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\\n'%(legend, auc_val, acc, low))

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")

def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data 
'''

# Save eval.py
with open('eval.py', 'w') as f:
    f.write(eval_py_content)
print("✅ Created eval.py")

# 2. Create options.py
options_py_content = '''import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="text-davinci-003", help="the model to attack")
        self.parser.add_argument('--ref_model', type=str, default="huggyllama/llama-7b")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--data', type=str, default="swj0419/WikiMIA", help="the dataset to evaluate")
        self.parser.add_argument('--length', type=int, default=64, help="the length of the input text")
        self.parser.add_argument('--key_name', type=str, default="input", help="the key name for input text") 
'''

with open('options.py', 'w') as f:
    f.write(options_py_content)
print("✅ Created options.py")

# 3. Create run.py (main script)
run_py_content = '''import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import openai
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from options import Options
from eval import *


def load_model(name1, name2):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)

    if "davinci" in name2:
        model2 = None
        tokenizer2 = None
    else:
        model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto')
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained(name2)
    return model1, model2, tokenizer1, tokenizer2

def calculatePerplexity_gpt3(prompt, modelname):
    prompt = prompt.replace('\\x00','')
    responses = None
    openai.api_key = "YOUR_API_KEY"
    while responses is None:
        try:
            responses = openai.Completion.create(
                        engine=modelname, 
                        prompt=prompt,
                        max_tokens=0,
                        temperature=1.0,
                        logprobs=5,
                        echo=True)
        except openai.error.InvalidRequestError:
            print("too long for openai API")
            return None, [], 0
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None, [], 0
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob)

 
def calculatePerplexity(sentence, model, tokenizer, gpu):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2):
    pred = {}

    if "davinci" in modelname1:
        result = calculatePerplexity_gpt3(text, modelname1)
        if result[0] is None:
            return None
        p1, all_prob, p1_likelihood = result
        result_lower = calculatePerplexity_gpt3(text.lower(), modelname1)
        if result_lower[0] is None:
            return None
        p_lower, _, p_lower_likelihood = result_lower
    else:
        p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
        p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    if "davinci" in modelname2:
        result_ref = calculatePerplexity_gpt3(text, modelname2)
        if result_ref[0] is None:
            return None
        p_ref, all_prob_ref, p_ref_likelihood = result_ref
    else:
        p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
    
    pred["ppl"] = p1
    pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1)/zlib_entropy
    
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    ex["pred"] = pred
    return ex

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, col_name, modelname1, modelname2):
    print(f"all data size: {len(test_data)}")
    all_output = []
    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2)
        if new_ex is not None:
            all_output.append(new_ex)
    return all_output


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    args.output_dir = f"{args.output_dir}/{args.target_model}_{args.ref_model}/{args.key_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model1, model2, tokenizer1, tokenizer2 = load_model(args.target_model, args.ref_model)
    if "jsonl" in args.data:
        data = load_jsonl(f"{args.data}")
    else:
        dataset = load_dataset(args.data, split=f"WikiMIA_length{args.length}")
        data = convert_huggingface_data_to_list_dic(dataset)

    all_output = evaluate_data(data, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model)
    fig_fpr_tpr(all_output, args.output_dir)
'''

with open('run.py', 'w') as f:
    f.write(run_py_content)
print("✅ Created run.py")

print(f"\nFiles created and ready for download:")
print("1. run.py - Main detection script")
print("2. eval.py - Evaluation functions") 
print("3. options.py - Command line options")
print("\nNext: Creating support files...")