import os
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, FedSA, FedSA_FLoRA, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
import json
import numpy as np

file_path = './HF_key.json'
with open(file_path, 'r') as file:
    keys = json.load(file)

datasets.utils.logging.set_verbosity_error()


def fl_finetune(
        # model/data params
        global_model: str = '',
        data_path: str = './data',
        alpha: float = 0.1,
        dev_data_path: str = './mmlu_test_1444.jsonl',
        output_dir: str = './lora-shepherd/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.5,
        num_communication_rounds: int = 50,
        num_clients: int = 10,
        # Local training hyperparams
        local_batch_size: int = 64,  # 64,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 10,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        strategy: str = 'FedSA',
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, f"{num_clients}_{alpha}")
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # model = LlamaForCausalLM.from_pretrained(
    #     global_model,
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map=device_map,
    # )

    # tokenizer = LlamaTokenizer.from_pretrained(global_model)

    if global_model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token=keys["hf_token"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token=keys["hf_token"],
        )

    if global_model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=keys["hf_token"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=keys["hf_token"])

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    acc_list = []
    acc_dict = {}


    for epoch in tqdm(range(num_communication_rounds)):
        
        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir, epoch)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()
           
            print("\nTerminating the local training of Client_{}".format(client_id))

            print('Client_{} has started evaluation'.format(client_id))

            acc = global_evaluation(client.model, tokenizer, prompter, dev_data_path)
            acc_dict[client_id] = (acc, epoch)
            print('Local Acc of Client_{} is:'.format(client_id), acc)

            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client
        print("Collecting the weights of clients and performing aggregation")
        model = FedSA(model, selected_clients_set,
                output_dir,
                local_dataset_len_dict,
                epoch,
                )

        

        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        print('Avg Local Acc of Epoch', str(epoch), 'is:', acc)
        acc_list.append(np.mean([acc for acc, e in acc_dict.values() if e == epoch]))


    # selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy)
    if strategy == 'FedSA_FLoRA':
        print("\nConducting the LoRA Stacking")
        model = FedSA_FLoRA(model,
                        previously_selected_clients_set,
                        output_dir,
                        local_dataset_len_dict, # So that we can average the weights
                        epoch= epoch, # This basacically to retrieve the directory of last selected clients
                        )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)
        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
        print('Global Accuracy after FedSA_FLoRA', str(epoch), 'is:', acc)
        acc_list.append(acc)

    print("Accuracy List:")
    print(acc_list)     
    print('Accuracy Dictionary')
    print(acc_dict)  

    #os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_challenge,mmlu --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
    filename = output_dir + 'log.txt'
    file = open(filename,'a')
    for i in range(len(acc_list)):
        s = str(acc_list[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("Log Saved")

    with open(output_dir + 'acc_dict.json', 'w') as f:
        json.dump(acc_dict, f)


if __name__ == "__main__":
    fire.Fire(fl_finetune)
