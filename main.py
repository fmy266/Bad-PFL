import torch, torchvision, argparse
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
from server import BasicServer
from client import BasicClient, PoisonClient
from fl_process import basic_fl_process
from event_emitter import *
from resnet import get_resnet
from utils import random_select, evaluate_accuracy, client_inner_dirichlet_partition, set_random_seed
from trigger import grid_trigger_adder
from random import shuffle
from pfl import *
from fba import *


def load_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--total_round", type=int, default=300)
    parser.add_argument("--model", type=str, default="resnet10")
    parser.add_argument("--model_size", type=int, default=10)
    parser.add_argument("--client_num", type=int, default=100)
    parser.add_argument("--bad_client_num", type=int, default=10)
    parser.add_argument("--select_client_num_per_round", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--client_dist", type=str, default="non_iid")
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--client_local_step", type=int, default=15)
    parser.add_argument("--client_batch", type=int, default=32)
    parser.add_argument("--pfl", type=str, default="fedbn")
    parser.add_argument("--ba", type=str, default="our")
    parser.add_argument("--ba_target_label", type=int, default=0)
    parser.add_argument("--ba_poison_rate", type=float, default=0.2)
    parser.add_argument("--ba_trigger_position", type=str, default="left_top")
    parser.add_argument("--agg_rule", type=str, default="avg")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = load_argument()

    #################################### env config ####################################

    if args.device == "cpu":
        device = torch.device(f"cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    set_random_seed(args.seed)

    #################################### FLconfig ####################################

    client_optimizer = partial(torch.optim.SGD, lr=args.learning_rate)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=transform)
    train_dataset_labels = train_dataset.targets
    test_dataset_labels = test_dataset.targets
    num_classes = 10
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    client_train_sample_nums = [int(len(train_dataset) / args.client_num) for _ in range(args.client_num)]
    client_test_sample_nums = [int(len(test_dataset) / args.client_num) for _ in range(args.client_num)]
    class_priors = np.random.dirichlet(alpha=[args.dir_alpha] * num_classes, size=args.client_num)
    client_train_data_indices = client_inner_dirichlet_partition(train_dataset_labels, args.client_num,
                                                                    num_classes=num_classes,
                                                                    dir_alpha=args.dir_alpha,
                                                                    client_sample_nums=client_train_sample_nums,
                                                                    class_priors=class_priors)
    client_test_data_indices = client_inner_dirichlet_partition(test_dataset_labels, args.client_num,
                                                                num_classes=num_classes,
                                                                dir_alpha=args.dir_alpha,
                                                                client_sample_nums=client_test_sample_nums,
                                                                class_priors=class_priors)
    client_train_dataloaders = [torch.utils.data.DataLoader(train_dataset, batch_size=args.client_batch,
                                                            sampler=SubsetRandomSampler(
                                                                client_train_data_indices[i]), drop_last=True)
                                for i in range(args.client_num)]
    client_test_dataloaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.client_batch,
                                                            sampler=SubsetRandomSampler(client_test_data_indices[i]), drop_last=True)
                                for i in range(args.client_num)]

    #################################### client config ####################################

    clients = []
    clients = [BasicClient(get_resnet(size=10, num_classes=num_classes).to(device), client_train_dataloaders[i], client_test_dataloaders[i],
                            torch.nn.CrossEntropyLoss(), client_optimizer) for i in range(args.client_num - args.bad_client_num)]
    clients.extend([PoisonClient(get_resnet(size=10, num_classes=num_classes).to(device), client_train_dataloaders[i],
                                    client_test_dataloaders[i], torch.nn.CrossEntropyLoss(), client_optimizer,
                                    poison_func=None)
                    for i in range(args.client_num - args.bad_client_num, args.client_num)])
    shuffle(clients)

    for idx, client in enumerate(clients):
        client.local_model.device = device
        client.cid = idx

    #################################### server config ####################################

    global_model = get_resnet(size=10, num_classes=num_classes)
    server = BasicServer(global_model.to(device))
    server.global_model.device = device
    server.agg_rule = args.agg_rule

    ############################### pfl config ###############################

    if args.pfl == "fedbn":
        use_fedbn(server)

    ############################### backdoor attack config ###############################

    
    if args.ba == "our":
        full_poison_func = use_our_attack(clients, server, args.ba_target_label, args.ba_poison_rate)
        
    
    ################################## run fl ##################################

    basic_fl_process(server, clients, local_steps=args.client_local_step, training_rounds=args.total_round,
                     select_rule=partial(random_select, nums=args.select_client_num_per_round))

    ################################## compute res ##################################

    acc_ls, asr_ls = [], []
    for client in clients:

        accuracy = evaluate_accuracy(client.local_model, client.test_dataloader)
        asr = evaluate_accuracy(client.local_model, client.test_dataloader, full_poison_func)
        print(f"Client id: {client.cid} \t Accuracy: {accuracy} \t ASR: {asr}")

        acc_ls.append(accuracy), asr_ls.append(asr)
    

    print(f"Avg acc: {torch.Tensor(acc_ls).mean().item():.2f}\tAcc std: {torch.Tensor(acc_ls).std().item():.2f}\t"
            f"Avg ASR: {torch.Tensor(asr_ls).mean().item():.2f}\tASR std: {torch.Tensor(acc_ls).std().item():.2f}")