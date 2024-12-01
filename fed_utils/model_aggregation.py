from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict
)
import torch
import os
from copy import deepcopy
from torch.nn.functional import normalize
from other import get_topk_mask, sparsify_model

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

def FedSA(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, flasc=False, dl_density=1.0, ul_density=1.0, l2_clip_norm=0.0, noise_multiplier=0.0):

    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)
    
    #global model parameters (only lora A)
    server_state_dict = get_peft_model_state_dict(model, adapter_name="default")
    server_params = {n: p.clone() for n, p in server_state_dict.items() if 'lora_A' in n}
    server_mask = {}
    
    #Apply download sparsification
    if flasc and dl_density < 1.0:
        all_params_flat = torch.cat([p.view(-1) for p in server_params.values()])
        server_mask_flat = get_topk_mask(all_params_flat.abs(), dl_density)
        curr = 0
        for n, p in server_params.items():
            numel = p.numel()
            mask = server_mask_flat[curr:curr + numel].view_as(p)
            server_mask[n] = mask
            server_params[n] = p * mask
            curr += numel
    else:
        for n in server_params:
            server_mask[n] = torch.ones_like(server_params[n])

    aggregate = None

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(
            output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin")
        
        client_weights = torch.load(single_output_dir)

        client_params = {n: p for n, p in client_weights.items() if 'lora_A' in n}
        
        neg_client_delta = {}
        for n in server_params.keys():
            if n in client_params:
                delta = server_params[n] - client_params[n]
                neg_client_delta[n] = delta
            else:
                print(f"Parameter {n} not found in client {client_id} weights.")
        
        if flasc and ul_density < 1.0:
            delta_flat = torch.cat([p.view(-1) for p in neg_client_delta.values()])
            delta_mask_flat = get_topk_mask(delta_flat.abs(), ul_density)
            curr = 0
            for n, p in neg_client_delta.items():
                numel = p.numel()
                mask = delta_mask_flat[curr:curr + numel].view_as(p)
                neg_client_delta[n] = p * mask
                curr += numel
        
        #DP clipping and noise addition
        delta_flat = torch.cat([p.view(-1) for p in neg_client_delta.values()])
        delta_norm = torch.norm(delta_flat).item()
        if l2_clip_norm > 0:
            divisor = max(delta_norm / l2_clip_norm, 1.0)
            for n in neg_client_delta:
                neg_client_delta[n] = neg_client_delta[n] / divisor
        
        for n in neg_client_delta:
            neg_client_delta[n] = neg_client_delta[n] * weights_array[k]
        
        if aggregate is None:
            aggregate = neg_client_delta
        else:
            for n in aggregate:
                aggregate[n] += neg_client_delta[n]
    
    #DP and noise addition to the aggregate
    if l2_clip_norm > 0:
        for n in aggregate:
            aggregate[n] = aggregate[n] / l2_clip_norm
    if noise_multiplier > 0:
        for n in aggregate:
            noise = noise_multiplier * torch.randn_like(aggregate[n])
            aggregate[n] += noise
    
    #Update server model
    updated_state_dict = server_state_dict.copy()
    for n in aggregate:
        updated_state_dict[n] = server_state_dict[n] - aggregate[n]
    
    set_peft_model_state_dict(model, updated_state_dict, adapter_name="default")
    
    return model