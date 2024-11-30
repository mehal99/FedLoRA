from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict
)
import torch
import os
from torch.nn.functional import normalize
from torch.nn import ZeroPad2d

def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, stacking, lora_r, heter, local_ranks, zero_padding, full):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    print("Weights:", weights_array)
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir, map_location = 'cpu')
        #print(single_weights)
        #print("y")
        x = 0
        if full:
            if k == 0:
                weighted_single_weights = single_weights
                for key in weighted_single_weights.keys():
                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
            else:
                for key in single_weights.keys():
                    weighted_single_weights[key] += single_weights[key] * (weights_array[k])
            
        else:
            if stacking:
                if zero_padding:
                    max_lora = max(local_ranks)
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                    else:
                        for key in single_weights.keys():
                            #print(single_weights[key].shape)
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                weighted_single_weights[key] += single_weights[key]
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                #print(single_weights[key][255,32])
                                weighted_single_weights[key] += single_weights[key]
                        
                else:
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            #weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k])
                            #print(weighted_single_weights[key].shape)
                            if heter:
                                x += 1
                                if weighted_single_weights[key].shape[0] == local_ranks[client_id]:
                                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)
                            else:
                                if weighted_single_weights[key].shape[0] == lora_r:
                                    weighted_single_weights[key] = weighted_single_weights[key] * (weights_array[k] * 1)

                    else:
                        for key in single_weights.keys():
                            if heter:
                                x += 1
                                if single_weights[key].shape[0] == local_ranks[client_id]:
                                    new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                    weighted_single_weights[key] = torch.cat(new, dim=0)
                            else:
                                if single_weights[key].shape[0] == lora_r:
                                    new = [weighted_single_weights[key], single_weights[key] * (weights_array[k]) * 1]
                                    weighted_single_weights[key] = torch.cat(new, dim=0)
                            
                            if heter:
                                if single_weights[key].shape[1] == local_ranks[client_id]:
                                    new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                    weighted_single_weights[key] = torch.cat(new, dim=1)
                            else:
                                if single_weights[key].shape[1] == lora_r:
                                    new = [weighted_single_weights[key], single_weights[key]]#  * (weights_array[k])]
                                    weighted_single_weights[key] = torch.cat(new, dim=1)

            else:
                if zero_padding:
                    max_lora = max(local_ranks)
                    if k == 0:
                        weighted_single_weights = single_weights
                        for key in weighted_single_weights.keys():
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                weighted_single_weights[key] = pad(weighted_single_weights[key]) * (weights_array[k])
                    else:
                        for key in single_weights.keys():
                            #print(single_weights[key].shape)
                            if single_weights[key].shape[0] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, 0, 0, max_lora-local_ranks[client_id]))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                weighted_single_weights[key] += single_weights[key]
                            elif single_weights[key].shape[1] == local_ranks[client_id]:
                                pad = ZeroPad2d(padding=(0, max_lora-local_ranks[client_id], 0, 0))
                                single_weights[key] = pad(single_weights[key]) * (weights_array[k])
                                #print(single_weights[key][255,32])
                                weighted_single_weights[key] += single_weights[key]
                else:
                    if k == 0:
                        weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                            single_weights.keys()}
                    else:
                        weighted_sindgle_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                            for key in
                                            single_weights.keys()}

    if stacking:
        torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        return model
    elif full:
        torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "pytorch_model.bin"))
        model.load_state_dict(weighted_single_weights)
        return model
    else:
        set_peft_model_state_dict(model, weighted_single_weights, "default")
        return model
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
