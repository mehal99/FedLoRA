import os
from typing import List
from tqdm import tqdm
import fire
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from transformers import ViTForImageClassification
from PIL import Image
from client_data_allocation import build_dataset
from fed_utils import eval_loop
import matplotlib.pyplot as plt

datasets.utils.logging.set_verbosity_error()

def plot(x, y, x_label, y_label, title, path):
    plt.figure()
    plt.plot(x, y, marker='o', label='Accuracy')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()
    print(f"Plot saved at: {path}")

def fl_finetune(
        # model/data params
        global_model: str = 'vit',
        data_path: str = './data',
        output_dir: str = './lora-shepherd/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.5,
        num_communication_rounds: int = 50,
        num_clients: int = 10,
        # Local training hyperparams
        local_batch_size: int = 64,  # 64,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 2,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "query", "value",
        ],
        ## heterogeneity params
        alpha: float = 0.1
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
            f"lora_target_modules: {lora_target_modules}\n",
            f"alpha: {alpha}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    # data_path = os.path.join(data_path, str(num_clients))
    # assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    if global_model == 'vit':
        global_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10).cuda()
   
    model = prepare_model_for_kbit_training(global_model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, config)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated classification has started..")

    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(alpha), str(num_clients), str(local_learning_rate))

    acc_list = []
    loss_list = []
    # def build_dataset(batch_size, n_clients, alpha=-1, seed=0):
    client_subsets, valloader, testloader, test_batch = build_dataset(local_batch_size, num_clients, alpha)
    
    for epoch in tqdm(range(num_communication_rounds)):
        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)    
        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, client_subsets[client_id], output_dir)
            client.train(local_learning_rate, local_num_epochs, test_batch)
            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict = client.terminate_local_training(epoch, local_dataset_len_dict)
            del client

        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch
                       )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        # acc = global_evaluation(model, processor, valloader)
        global_acc, global_loss = eval_loop(model, testloader)
        print('Acc of Epoch', str(epoch), 'is:', global_acc)
        print('Loss of Epoch', str(epoch), 'is:', global_loss)
        acc_list.append(global_acc)
        loss_list.append(global_loss)

    print(f"Accuracy of the global model across epochs: {acc_list}")
    print(f"Loss of the global model across epochs: {loss_list}")

    accuracy_plot_path = os.path.join(output_dir, 'accuracy_plot.png')
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    acc_percentage = [acc * 100 for acc in acc_list]
    plot(x=range(num_communication_rounds), y=acc_percentage, x_label="Communication Rounds", y_label="Accuracy (%)", title="Global Model Accuracy vs. Communication Rounds", path=accuracy_plot_path)
    plot(x=range(num_communication_rounds), y=loss_list, x_label="Communication Rounds", y_label="Loss", title="Global Model Loss vs. Communication Rounds", path=loss_plot_path)

    #os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_challenge,mmlu --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
    filename = output_dir + 'log.txt'
    file = open(filename,'a')
    s = "Accuracy"
    for i in range(len(acc_list)):
        s += str(acc_list[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    s = "Loss"
    for i in range(len(loss_list)):
        s += str(loss_list[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("Log Saved")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
