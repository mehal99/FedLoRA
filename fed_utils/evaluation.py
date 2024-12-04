import os
from typing import List
from tqdm import tqdm
import fire
import torch
import datasets
from transformers import GenerationConfig
import json
import csv
from peft import set_peft_model_state_dict
import numpy as np
import random

model_type = 'llama'
datasets.utils.logging.set_verbosity_error()
device_map = "auto"
max_new_token: int = 32
verbose: bool = False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)

def global_evaluation(model, tokenizer, prompter, dev_data_path):
    data_class =  ['high_school_chemistry', 'high_school_geography', 'econometrics', 'philosophy', 'high_school_world_history', 'college_physics', 'logical_fallacies', 'college_chemistry', 'abstract_algebra', 'college_computer_science', 'high_school_european_history', 'security_studies', 'moral_scenarios', 'high_school_mathematics', 'marketing', 'college_mathematics', 'professional_medicine', 'high_school_computer_science', 'high_school_statistics', 'jurisprudence', 'moral_disputes', 'high_school_physics', 'college_biology', 'electrical_engineering', 'virology', 'management', 'formal_logic', 'high_school_biology', 'prehistory', 'high_school_psychology', 'professional_accounting']
    right_count_dict = dict.fromkeys(data_class, 0)
    total_count_dict = dict.fromkeys(data_class, 0)
    acc_count_dict = dict.fromkeys(data_class, 0)
    with open(dev_data_path, 'r') as f:
        test_set = json.load(f)
    count=0

    if model_type == 'llama':
        sampling = GenerationConfig(
            do_sample=True,
            temperature=0.2,
            top_p=0.6,
            top_k=30,
            num_beams=1,
            max_new_tokens=max_new_token,
            early_stopping=True,
        )

    if model_type == 'gpt2':
        sampling = GenerationConfig(
            bos_token_id = 50256,
            eos_token_id = 50256,
            _from_model_config = True,
        )

    for data_point in tqdm(test_set):
        count +=1
        target = data_point["output"]
        class_test_set = data_point["class"]
        
        tgt_ans_idx = target.replace('The answer is: ','').split('. ')[0]
        tgt_ans = target.replace('The answer is: ','').split('. ')[1]

        test_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            'The answer is: ',
        )

        with torch.autocast("cuda"):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            input =inputs["input_ids"].to('cuda')
            with torch.no_grad():
                #print(tokenizer.eos_token_id, tokenizer.pad_token_id)
                generation_output = model.generate(
                    input_ids=input,
                    generation_config=sampling,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_token,
                    pad_token_id=tokenizer.eos_token_id
                )
            generation_output_decoded = tokenizer.decode(generation_output.sequences[0])
            # print(generation_output_decoded)
            split = prompter.template["response_split"]
            ans = generation_output_decoded.split(split)[-1].strip()
            if verbose:
                print('-------------------')
                print(test_prompt)
                print(tgt_ans)
                print(tgt_ans_idx)
                print(ans)
            if tgt_ans_idx+'.' in ans or tgt_ans in ans:
            # if tgt_ans_idx in ans or tgt_ans in ans:
                right_count_dict[class_test_set] += 1
            total_count_dict[class_test_set] += 1

    mean_acc = 0.

    for key in acc_count_dict.keys():
        tmp = right_count_dict[key]/total_count_dict[key]
        mean_acc += tmp
        acc_count_dict[key] = tmp
    mean_acc /= len(acc_count_dict.keys())
    csv_data = [right_count_dict, total_count_dict, acc_count_dict]

    if verbose:
       print(right_count_dict)
    #print(total_count_dict)
    print('Acc: ', acc_count_dict)
    print()
    #score = eval_usmle(model, dev_data_path, tokenizer, verbose=False)
    print('========== Accuracy ==========')
    print(mean_acc)
    
    return mean_acc