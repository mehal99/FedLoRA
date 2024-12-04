from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict
)
import torch
import os
from torch.nn.functional import normalize
from torch import nn


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

def FedSA_old(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):

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




def FedSA(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):

    weights_array = torch.nn.functional.normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    aggregated_A = {}

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(
            output_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin")

        # Load client's PEFT state dict
        single_weights = torch.load(single_output_dir, map_location='cpu')

        # Extract only 'lora_A' matrices
        A_state_dict = {key: value.clone() for key, value in single_weights.items() if 'lora_A' in key}

        # Adjust keys to match the expected format
        A_state_dict = {key.replace('.lora_A.weight', '.lora_A.default.weight'): value for key, value in A_state_dict.items()}

        for key in A_state_dict.keys():
            A_state_dict[key] *= weights_array[k]

        if k == 0:
            # Initialize
            aggregated_A = A_state_dict
        else:
            # Add the weighted LoRA A matrices from subsequent clients to the aggregated_A
            for key in aggregated_A.keys():
                aggregated_A[key] += A_state_dict[key]

    # Retrieve the global PEFT state dict
    global_peft_state_dict = get_peft_model_state_dict(model, adapter_name="default")

    # Update only the 'lora_A' matrices in the global PEFT state dict
    for key in aggregated_A.keys():
        if key in global_peft_state_dict:
            global_peft_state_dict[key] = aggregated_A[key].clone()
        else:
            print(f"Warning: {key} not found in global PEFT state dict.")

    # Set the updated PEFT state dict back to the model
    set_peft_model_state_dict(model, global_peft_state_dict, adapter_name="default")

    # Save the aggregated lora_A for clients to load
    aggregated_lora_A_path = os.path.join(output_dir, str(epoch), 'aggregated_lora_A.pth')
    os.makedirs(os.path.dirname(aggregated_lora_A_path), exist_ok=True)
    torch.save(aggregated_A, aggregated_lora_A_path)

    print(f"Aggregated `lora_A` and updated global model for epoch {epoch}.")

    return model

def FedSA_FLoRA(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):

    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    client_A_matrices = []
    client_B_matrices = []


    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(
            output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin")
        
        single_weights = torch.load(single_output_dir)

        A_state_dict = {key: value.clone() for key, value in single_weights.items() if 'lora_A' in key}
        B_state_dict = {key: value.clone() for key, value in single_weights.items() if 'lora_B' in key}

        client_A_matrices.append(A_state_dict)
        client_B_matrices.append(B_state_dict)

        # Average lora_A
        # for key in A_state_dict.keys():
        #     A_state_dict[key] *= weights_array[k]

    # average of 'lora_A' parameters
    averaged_A = {}
    A_keys = client_A_matrices[0].keys()
    for key in A_keys:
        averaged_A[key] = sum([client_A_matrices[k][key] for k in range(len(selected_clients_set))])

    # Stacking duplicates of averaged lora_A parameters
    duplicated_A = {}
    for key in averaged_A.keys():
        avg_matrix = averaged_A[key]
        duplicated_matrices = [avg_matrix] * len(selected_clients_set)
        duplicated_A[key] = torch.cat(duplicated_matrices, dim=0)

    # Stacking clients lora_B parameters
    stacked_B = {}
    B_keys = client_B_matrices[0].keys()
    for key in B_keys:
        stacked_B[key] = torch.cat([client_B_matrices[k][key] for k in range(len(selected_clients_set))], dim=1)

    # current rank
    r = model.peft_config["default"].r  
    # number of clients
    K = len(selected_clients_set)
    # the new stacked rank
    new_r = K * r

    from peft.tuners.lora import LoraLayer

    # Get current LoRA 
    lora_config = model.peft_config["default"]

    # Update the rank in the configuration
    lora_config.r = new_r

    # Rebuild the LoRA layers in the model
    # for module_name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         # Get the original in_features and out_features
    #         in_features = module.in_features
    #         out_features = module.out_features

    #         # Reinitialize the LoRA layers with the new rank
    #         module.r = new_r
    #         module.lora_A = torch.nn.Parameter(
    #             module.lora_A.new_zeros((new_r, in_features))
    #         )
    #         module.lora_B = torch.nn.Parameter(
    #             module.lora_B.new_zeros((out_features, new_r))
    #         )
    #         module.scaling = lora_config.lora_alpha / new_r
    for module_name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            in_features = module.in_features
            out_features = module.out_features

            # Update rank
            module.r = new_r
            # print('Reinitializing LoraLayer:')
            # print(module)

            # Reinitialize lora_A and lora_B within the ModuleDict
            # Assuming 'default' is the key used in ModuleDict
            # Update lora_A
            if 'default' in module.lora_A:
                module.lora_A['default'] = torch.nn.Linear(in_features, new_r, bias=False, device='cuda:0')
                torch.nn.init.zeros_(module.lora_A['default'].weight)
            else:
                raise KeyError(f"'default' key not found in lora_A of module {module_name}")

            # Update lora_B
            if 'default' in module.lora_B:
                module.lora_B['default'] = torch.nn.Linear(new_r, out_features, bias=False, device='cuda:0')
                torch.nn.init.zeros_(module.lora_B['default'].weight)
            else:
                raise KeyError(f"'default' key not found in lora_B of module {module_name}")




            # print('After Reinitializing LoraLayer:')
            # print(module)



    # global_peft_state_dict = model.get_peft_model_state_dict()
    global_peft_state_dict = get_peft_model_state_dict(model, adapter_name="default")

    # Apply the dublicated averaged lora_A on the server model's lora
    for key in duplicated_A.keys():
        global_peft_state_dict[key] = duplicated_A[key]

    # Apply the stacked lora_B
    for key in stacked_B.keys():
        global_peft_state_dict[key] = stacked_B[key]
    print(model)
    set_peft_model_state_dict(model, global_peft_state_dict, adapter_name="default")

    return model