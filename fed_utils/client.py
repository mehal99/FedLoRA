import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from copy import deepcopy
from tqdm import tqdm

class GeneralClient:
    def __init__(self, client_id, model, client_subset, output_dir):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.local_data = client_subset
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))


    def train(self, client_lr, client_epochs, test_batch):
        client_opt = torch.optim.SGD(self.model.parameters(), lr=client_lr, momentum=0.9)
        for _ in tqdm(range(client_epochs)):
            for x,y in self.local_data:
                x, y = x.to("cuda"), y.to("cuda")
                loss, correct = test_batch(self.model, x, y)
                client_opt.zero_grad()
                loss.backward()
                client_opt.step()


    def terminate_local_training(self, epoch, local_dataset_len_dict):

        local_dataset_len_dict[self.client_id] = len(self.local_data)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        return self.model, local_dataset_len_dict
