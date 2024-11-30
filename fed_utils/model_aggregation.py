from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict
)
import torch
import os
from torch.nn.functional import normalize


def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    set_peft_model_state_dict(model, weighted_single_weights, "default")

    return model

def FedSA(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):

    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    aggregated_A = {}

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(
            output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin")
        
        single_weights = torch.load(single_output_dir)

        A_state_dict = {key: value.clone() for key, value in single_weights.items() if 'lora_A' in key}

        # Averaging
        for key in A_state_dict.keys():
            A_state_dict[key] *= weights_array[k]

        # client 0
        if k == 0:
            # Initialization
            aggregated_A = A_state_dict
        else:
            # Add the weighted LoRA A matrices from subsequent clients to the aggregated_A (Averaged)
            for key in aggregated_A.keys():
                aggregated_A[key] += A_state_dict[key]

    # global_peft_state_dict = model.get_peft_model_state_dict()
    global_peft_state_dict = get_peft_model_state_dict(model, adapter_name="default")

# 
    for key in aggregated_A.keys():
        global_peft_state_dict[key] = aggregated_A[key]


    set_peft_model_state_dict(model, global_peft_state_dict, adapter_name="default")

    return model