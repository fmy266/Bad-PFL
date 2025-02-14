import torch
import numpy as np
import pandas as pd
from collections import Counter
from functools import partial


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_mix_precision(func):
    def wrapper(*args, **kwargs):
        with torch.cuda.amp.autocast():
            return func(*args, **kwargs)
    return wrapper


def random_select(server, clients, nums):
    assert isinstance(nums, (int, float))
    nums = int(nums * len(clients)) if isinstance(nums, float) else nums
    return torch.randperm(len(clients))[:nums].tolist()


def evaluate_accuracy(model, datloader, data_transform=None):
    device = model.device
    correct, total = 0, 0

    model.eval()
    for inputs, labels in datloader:
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(device), labels.to(device)

        if data_transform is not None:
            inputs, labels = data_transform(inputs, labels)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        torch.cuda.empty_cache()

    accuracy = 100 * correct / total
    return accuracy


def client_inner_dirichlet_partition(targets, num_clients, num_classes, dir_alpha, client_sample_nums, class_priors=None):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    if class_priors is None:
        class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                           size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            if class_amount[curr_class] <= 0:
                while True:
                    new_class = np.random.randint(num_classes)
                    if class_amount[new_class] > 0:
                        curr_class = new_class
                        break
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict


def partition_report(targets, data_indices, class_num=None, verbose=True, file=None):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    if not class_num:
        class_num = max(targets) + 1

    sorted_cid = sorted(data_indices.keys())

    stats_rows = []
    for client_id in sorted_cid:
        indices = data_indices[client_id]
        client_targets = targets[indices]
        client_sample_num = len(indices)
        client_target_cnt = Counter(client_targets)
        cur_client_stat = {'cid': client_id}
        for cls in range(class_num):
            cur_client_stat[f'class-{cls}'] = client_target_cnt[cls] if cls in client_target_cnt else 0
        cur_client_stat['TotalAmount'] = client_sample_num
        stats_rows.append(cur_client_stat)

    stats_df = pd.DataFrame(stats_rows)
    if file is not None:
        stats_df.to_csv(file, header=True, index=False)
    if verbose:
        print("Class sample statistics:")
        print(stats_df)

    return stats_df
