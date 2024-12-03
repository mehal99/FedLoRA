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


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, epoch):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, f"local_training_{self.client_id}.json")
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", f"local_output_{self.client_id}")
        
        # Load aggregated lora_A
        aggregated_lora_A_path = os.path.join(output_dir, str(epoch), 'aggregated_lora_A.pth')
        if os.path.exists(aggregated_lora_A_path):
            aggregated_A = torch.load(aggregated_lora_A_path)
        else:
            # First epoch: use model's current lora_A
            aggregated_A = get_peft_model_state_dict(self.model, adapter_name="default")
            aggregated_A = {key: value.clone() for key, value in aggregated_A.items() if 'lora_A' in key}
        
        # Load client's own lora_B
        if epoch > 0:
            client_lora_B_path = os.path.join(output_dir, str(epoch-1), f"local_output_{self.client_id}", 'lora_B.pth')
            if os.path.exists(client_lora_B_path):
                client_B = torch.load(client_lora_B_path)
            else:
                print(f"Client {self.client_id} lora_B not found for epoch {epoch-1}, using initial lora_B.")
                client_B = get_peft_model_state_dict(self.model, adapter_name="default")
                client_B = {key: value.clone() for key, value in client_B.items() if 'lora_B' in key}
        else:
            # First epoch: use model's current lora_B
            client_B = get_peft_model_state_dict(self.model, adapter_name="default")
            client_B = {key: value.clone() for key, value in client_B.items() if 'lora_B' in key}
        
        # Combine aggregated lora_A and client's own lora_B
        peft_state_dict = {}
        peft_state_dict.update(aggregated_A)
        peft_state_dict.update(client_B)
        
        # Set the combined PEFT state dict into the model
        set_peft_model_state_dict(self.model, peft_state_dict, adapter_name="default")

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                  ),
                                                  )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = get_peft_model_state_dict(self.model, adapter_name="default")
        single_output_dir = os.path.join(self.output_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, os.path.join(single_output_dir, "pytorch_model.bin"))
        
        # Save client's own lora_B
        B_state_dict = {key: value.clone() for key, value in new_adapter_weight.items() if 'lora_B' in key}
        torch.save(B_state_dict, os.path.join(single_output_dir, 'lora_B.pth'))
        
        # Reset model to previous state if needed
        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id