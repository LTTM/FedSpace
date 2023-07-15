import os
import logging
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets_CL import CIFAR100_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar100_train():
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    return train_transform


def _data_transforms_cifar100_test():
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return test_transform


def load_cifar100_data(datadir):
    train_transform = _data_transforms_cifar100_train()
    test_transform = _data_transforms_cifar100_test()

    cifar10_train_ds = CIFAR100_truncated(datadir, classes=None, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR100_truncated(datadir, classes=None, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    net_dataidx_map = dict()

    if partition == "powerlaw-dirichlet":
        num_classes = 100
        samples_per_class = 500
        esponential_factor = 3
        cifar100_size = y_train.shape[0]

        # powerlaw distribution for the number of samples of each client
        samples = n_nets
        s = np.random.power(esponential_factor, samples)
        num_samples_per_client = ((s / s.sum()) * cifar100_size)
        num_samples_per_client = [round(n) for n in num_samples_per_client]
        remainder = np.asarray(num_samples_per_client).sum() - cifar100_size
        if remainder != 0:
            num_samples_per_client = manage_remainder(remainder, n_nets, num_samples_per_client)

        print("NUM_SAMPLES_PER_CLIENT: ", num_samples_per_client)

        # dirichlet distribution for the number of classes of each client
        idx_k = [np.where(y_train == label)[0] for label in range(num_classes)]
        for label in range(num_classes):
            np.random.shuffle(idx_k[label])

        # initialize variables
        proportions_clients = [np.zeros((num_classes,), dtype=int) for _ in range(n_nets)]
        samples_availability = np.full((num_classes,), samples_per_class)
        available_classes = np.ones((num_classes,), dtype=int)
        indices_available_classes = np.asarray(range(num_classes, 2 * num_classes)) * available_classes
        sampling_classes = num_classes

        for client in range(n_nets):
            proportions = np.zeros((num_classes,))
            probability = np.random.dirichlet(np.repeat(alpha, sampling_classes))
            proportions[indices_available_classes - num_classes] += probability
            proportions = np.asarray(
                [round(proportions[label] * num_samples_per_client[client]) for label in range(num_classes)])

            # check whether the current client uses more than 5000 samples for class
            violations_class_samples = np.where(proportions > samples_per_class)[0]
            while violations_class_samples.size > 0:
                proportions[violations_class_samples] -= 1
                violations_class_samples = np.where(proportions > samples_per_class)[0]

            # check whether the current client uses more samples than the remaining ones
            violations_remaining_samples = np.where(proportions > samples_availability)[0]
            proportions[violations_remaining_samples] = samples_availability[violations_remaining_samples]

            # update variables for the next client
            samples_availability -= proportions
            out_of_stock_classes = np.where(samples_availability == 0)[0]
            available_classes[out_of_stock_classes] = 0
            sampling_classes = num_classes - out_of_stock_classes.size
            indices_available_classes = np.asarray(range(num_classes, 2 * num_classes)) * available_classes
            indices_available_classes = indices_available_classes[indices_available_classes != 0]

            # store the class distribution of the current client
            proportions_clients[client] += proportions

        proportions_clients = [proportions.tolist() for proportions in proportions_clients]

        # distribute remaining samples
        distribute_remaining_samples(n_nets, num_samples_per_client, proportions_clients, samples_availability)

        # assign samples to clients according to distributions computed above
        idx_batch = assign_samples(num_classes, n_nets, proportions_clients, idx_k)

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def manage_remainder(remainder, indexes, data_count_array):
    random_clients_idx = np.random.randint(indexes, size=abs(remainder))
    data_count_array = np.asarray(data_count_array)

    if remainder > 0:
        zero_entries = np.where(data_count_array[random_clients_idx] == 0)[0]
        while len(zero_entries) > 0:
            random_clients_idx = np.random.randint(indexes, size=abs(remainder))
            zero_entries = np.where(data_count_array[random_clients_idx] == 0)[0]

        # we subtract an example from random elements of data_count_array for remainder times
        data_count_array[random_clients_idx] = data_count_array[random_clients_idx] - 1

    elif remainder < 0:
        # we add an example from random elements of data_count_array for remainder times
        data_count_array[random_clients_idx] = data_count_array[random_clients_idx] + 1

    return data_count_array.tolist()


# assign samples to clients according to distributions computed above
def assign_samples(num_classes, n_nets, proportions_clients, idx_k):

    idx_batch = [[] for _ in range(n_nets)]
    for label in range(num_classes):
        class_samples_distribution = [proportions_clients[client][label] for client in range(n_nets)]
        proportions = np.cumsum(class_samples_distribution).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k[label], proportions))]

    return idx_batch


# distribute remaining samples
def distribute_remaining_samples(n_nets, num_samples_per_client, proportions_clients, samples_availability):

    remaining_classes_indexes = np.where(samples_availability > 0)[0]
    for client in range(n_nets):
        while np.asarray(proportions_clients[client]).sum() < num_samples_per_client[client] and remaining_classes_indexes.size > 0:
            index = remaining_classes_indexes[0]
            proportions_clients[client][index] = proportions_clients[client][index] + 1
            samples_availability[remaining_classes_indexes[0]] -= 1
            remaining_classes_indexes = np.where(samples_availability > 0)[0]


# for centralized training
def get_dataloader_train(datadir, train_bs, classes=None, dataidxs=None):
    return get_dataloader_CIFAR100_train(datadir, train_bs, classes, dataidxs)


def get_dataloader_test(datadir, test_bs, classes=None):
    return get_dataloader_CIFAR100_test(datadir, test_bs, classes)


def get_dataloader_CIFAR100_train(datadir, train_bs, classes=None, dataidxs=None):
    dl_obj = CIFAR100_truncated

    transform_train = _data_transforms_cifar100_train()

    train_ds = dl_obj(datadir, classes, dataidxs=dataidxs, train=True, transform=transform_train, download=True, bs=train_bs)
    if len(train_ds) == 0:
        return None, 0

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)

    if classes is not None:
        return train_dl, len(train_ds.indexes_task)
    else:
        return train_dl


def get_dataloader_CIFAR100_test(datadir, test_bs, classes=None):
    dl_obj = CIFAR100_truncated

    transform_test = _data_transforms_cifar100_test()

    test_ds = dl_obj(datadir, classes, dataidxs=None, train=False, transform=transform_test, download=True)

    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return test_dl


def load_partition_data_cifar100(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, num_classes_first_task, task_num, save_distribution_file, one_task):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))

    # get local dataset
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    train_data_task_dict = dict()

    # divide the classes among all tasks
    task_size = 1 if one_task else int((class_num - num_classes_first_task) / task_num)

    task_classes = {0: list(range(num_classes_first_task))}
    first_new_class = num_classes_first_task
    for task in range(1, task_num + 1):
        task_classes[task] = list(range(first_new_class, first_new_class + task_size))
        first_new_class = first_new_class + task_size

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, len(dataidxs)))

        train_data_local = {}
        train_data_local_num = {}

        for task in range(0, task_num + 1):
            print("Fetching data for task {} of client {}...".format(task, client_idx))
            train_data_local[task], train_data_local_num[task] = get_dataloader_train(data_dir, batch_size, task_classes[task], dataidxs)

        train_data_local_dict[client_idx] = train_data_local
        train_data_local_num_dict[client_idx] = train_data_local_num

    for task in range(task_num + 1):
        train_data_task_dict[task], _ = get_dataloader_train(data_dir, batch_size, task_classes[task])

    if save_distribution_file != '':
        path = 'dataset_splits'
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(os.path.join(path, save_distribution_file), 'wb') as file:
            pickle.dump(train_data_local_num_dict, file)
            pickle.dump(train_data_local_dict, file)
            pickle.dump(train_data_task_dict, file)
            pickle.dump(class_num, file)
            pickle.dump(task_classes, file)
            file.close()

    return train_data_local_num_dict, train_data_local_dict, train_data_task_dict, class_num, task_classes


def load_distribution_from_file(load_distribution_file):
    with open(os.path.join('dataset_splits', load_distribution_file), 'rb') as file:
        train_data_local_num_dict = pickle.load(file)
        train_data_local_dict = pickle.load(file)
        train_data_task_dict = pickle.load(file)
        class_num = pickle.load(file)
        task_classes = pickle.load(file)
        file.close()

    return train_data_local_num_dict, train_data_local_dict, train_data_task_dict, class_num, task_classes