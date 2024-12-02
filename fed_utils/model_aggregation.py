from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict
)
import torch
import os
from copy import deepcopy
from torch.nn.functional import normalize
from fed_utils.other import get_topk_mask, sparsify_model

def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, flasc=False, dl_density=1.0, ul_density=1.0, l2_clip_norm=0.0, noise_multiplier=0.0):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    client_model = deepcopy(model)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        client_weights = torch.load(single_output_dir)
        client_model.load_state_dict(client_weights)

        if flasc and ul_density < 1.0:
            sparsify_model(client_model, ul_density)

        sparse_client_weights = client_model.state_dict()

        #DP clipping and noise addition
        sparse_client_flat = torch.cat([value.view(-1) for value in sparse_client_weights.values()])
        l2_norm = torch.norm(sparse_client_flat, p=2).item()
        if l2_clip_norm > 0.0:
            divisor = max(l2_norm / l2_clip_norm, 1.0)
            for n in sparse_client_weights:
                sparse_client_weights[n] /= divisor

        if k == 0:
            weighted_sparse_client_weights = {key: sparse_client_weights[key] * (weights_array[k]) for key in
                                       sparse_client_weights.keys()}
        else:
            weighted_sparse_client_weights = {key: weighted_sparse_client_weights[key] + sparse_client_weights[key] * (weights_array[k])
                                       for key in
                                       sparse_client_weights.keys()}

    #DP and noise addition to the client aggregate
    if l2_clip_norm > 0.0:
        for n in weighted_sparse_client_weights:
            weighted_sparse_client_weights[n] /= l2_clip_norm
    if noise_multiplier > 0.0:
        for val in weighted_sparse_client_weights:
            noise = noise_multiplier * torch.randn_like(weighted_sparse_client_weights[val])
            weighted_sparse_client_weights[val] += noise
    
    set_peft_model_state_dict(model, weighted_sparse_client_weights, adapter_name="default")
 
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

    for key in aggregated_A.keys():
        global_peft_state_dict[key] = aggregated_A[key]


    set_peft_model_state_dict(model, global_peft_state_dict, adapter_name="default")

    return model