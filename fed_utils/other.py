import torch
from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict
)

def other_function():

    return print("design the other functions you need")

def get_topk_mask(x, density):
    k = int(x.numel() * density)
    if k == 0:
        return torch.zeros_like(x)
    else:
        _, idx = torch.topk(x.abs().view(-1), k)
        mask = torch.zeros_like(x.view(-1))
        mask[idx] = 1
        return mask.view_as(x)

def sparsify_model(model, density):
    model_state_dict = get_peft_model_state_dict(model, adapter_name="default")
    model_params = {n: p.clone() for n, p in model_state_dict.items()}
    all_params_flat = torch.cat([p.view(-1) for p in model_params.values()])
    mask_flat = get_topk_mask(all_params_flat, density)
    start = 0
    for key, value in model_params.items():
        numel = value.numel()
        model_params[key] = value * mask_flat[start:start+numel].view_as(value)
        start += numel
    set_peft_model_state_dict(model, model_params, adapter_name="default")
