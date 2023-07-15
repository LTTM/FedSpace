import argparse
import logging
import random
import numpy as np
import torch

from data.data_loader_CL import load_partition_data_cifar100
from data.data_loader_CL import load_distribution_from_file

from models.mobilenet_PASS import mobilenet
from models.ResNet_18_PASS import resnet_18

from fedavg_api_CL import FedAvgAPI
from client.my_model_trainer_classification_CL import MyModelTrainer as MyModelTrainerCLS
import wandb
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--name', type=str, default="", help="name of the simulation")
    parser.add_argument('--stop_at_round', default=500, type=int, help='stop training after stop_at_round rounds')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                        help='if you want wandb offline set to True, otherwise it uploads results on cloud')
    parser.add_argument('--wandb_entity', type=str, default='', help='set the name of your wandb entity to log the results')
    parser.add_argument('--same_order', action='store_true', default=False,
                        help="force clients to learn tasks in the same order")
    parser.add_argument('--same_time', action='store_true', default=False,
                        help="force clients to learn tasks at the same time")
    parser.add_argument('--mask_model', action='store_true', default=False,
                        help="mask the model output")
    parser.add_argument('--more_asynch', action='store_true', default=False,
                        help="a bit more asynchronous")
    parser.add_argument('--aggregate_with_global_model', action='store_true', default=False,
                        help="average parameter after fedavg with the previous global model")
    parser.add_argument('--global_weight', default=0.5, type=float,
                        help="weight of the global model")
    parser.add_argument('--centralized_fractal_pretrain_steps', type=int, default=0,
                        help="number of centralized fractal pretrain")
    parser.add_argument('--fractal_pretrain_rounds', type=int, default=0,
                        help="number of federated fractal pretraining rounds")
    parser.add_argument('--federated_fractal_pretrain_steps', type=int, default=10,
                        help="number of iterations to be performed in each round")
    parser.add_argument('--end_task_from_round_number', action='store_true', default=False,
                        help="end task when current round > round task-1 without considering only the client rounds")
    parser.add_argument('--aggregate_proto', action='store_true', default=False,
                        help="aggregate local prototypes into global ones")
    parser.add_argument('--aggr_proto_step', type=int, default=1, help="aggregate proto every n rounds")
    parser.add_argument('--aggr_proto_after_round', type=int, default=1, help="global proto after n rounds")
    parser.add_argument('--update_teacher_step', default=1, type=int,
                        help='number of rounds after the teacher model is updated')
    parser.add_argument('--update_teacher_ema', default=1., type=float,
                        help="exponential moving average smoothing factor for KD teacher's params [ema*new + (1-ema)*old]")

    parser.add_argument('--model', type=str, default='resnet_18', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--data_dir', type=str, default='cifar100',
                        help='data directory')
    parser.add_argument('--partition_method', type=str, default='powerlaw-dirichlet', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=1, metavar='PA',
                        help='partition alpha (default: 1)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--step_size', type=int, default=200, help='multiply lr by 0.1 after step_size epochs')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')
    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')
    parser.add_argument('--setup', type=str, default='federated', help='training setup')
    parser.add_argument('--comm_rounds', type=int, default=500,
                        help="cumulative communication rounds")
    parser.add_argument('--force_rounds_per_task', type=int, default=-1,
                        help="force each task to have force_rounds_per_task rounds")
    parser.add_argument('--repeat_tasks_n_times', type=int, default=0,
                        help="after the tasks are finished, restart from the first task")

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--save_distribution_file', type=str, default='',
                        help='filename of the .txt file in which the dataset distribution will be saved')
    parser.add_argument('--load_distribution_file', type=str, default='',
                        help='filename of the .txt file containing the dataset distribution to be loaded')

    parser.add_argument('--proto_loss_after_first_task', action='store_true', default=False,
                        help="when there is only one client (centralized) whether to activate the pass loss only after"
                             "the first task is finished")

    parser.add_argument('--test_every_n_rounds', type=int, default=1, help='perform test every n rounds')
    parser.add_argument('--test_clients_every_n_rounds', type=int, default=10e5, help='perform test every n rounds')

    parser.add_argument('--num_classes_first_task', type=int, default=10,
                        help='the number of classes in the first task')
    parser.add_argument('--task_num', type=int, default=9,
                        help='the number of incremental steps')
    parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
    parser.add_argument('--width_multiplier', default=1, type=int, help='width multiplier factor for mobilenet')
    parser.add_argument('--temp', default=0.1, type=float, help='training time temperature')
    parser.add_argument('--lambda_proto_aug', default=0, type=float, help='protoAug loss weight')
    parser.add_argument('--lambda_repr_loss', default=0, type=float, help='representation loss weight')
    parser.add_argument('--repr_loss_temp', default=1., type=float, help='representation loss temp')
    parser.add_argument('--repr_loss_wo_aug', default=True, type=str2bool, help='representation loss w/o augmented new-class samples')

    parser.add_argument('--proto_queue_length', default=100, type=int, help='length of the proto queue')
    parser.add_argument('--proto_queue',  action='store_true', default=False, help="use proto_queue for proto_aug loss")
    parser.add_argument('--mean_proto_queue',  action='store_true', default=False,
                        help="compute global prototypes as the weighted mean of the proto_queue")
    parser.add_argument('--multi_radius',  action='store_true', default=False,
                        help="keep multiple radiuses, one for each entry of the queue")

    parser.add_argument('--aggregate_mean_radius',  action='store_true', default=False,
                        help="compute the mean radius as non weighted mean")

    parser.add_argument('--aggregate_proto_by_class', action='store_true', default=False,
                        help="aggr global proto by class")

    parser.add_argument('--one_task', action='store_true', default=False, help="force one class only")

    parser.add_argument('--n_rounds_scheduling', default=10000, type=int, help='number of rounds of scheduling step')
    parser.add_argument('--lr_scheduler_multiplier', default=0.5, type=float,
                        help='scheduler multiplier after n_rounds_scheduling')

    # ----------- PROTO PARAM -----------------------
    parser.add_argument('--location_proto_aug', default='local', type=str,
                        help="local or global, which prototypes to use for the proto aug loss")
    parser.add_argument('--proto_loss_curr_classes', default=False, type=str2bool,
                        help="use all classes on proto loss, even current ones")
    parser.add_argument('--ema_global', default=0.9, type=float, help='exponential moving average smoothing factor')
    parser.add_argument('--folder', default='', type=str,
                        help='folder in which save prototypes')

    return parser

def load_training_data(args, dataset_name):

    # load and distribute training data for CIFAR100
    if args.load_distribution_file != '':
        train_data_local_num_dict, train_data_local_dict, train_data_task_dict, \
        class_num, task_classes = load_distribution_from_file(args.load_distribution_file)

    else:
        train_data_local_num_dict, train_data_local_dict, train_data_task_dict,\
        class_num, task_classes = load_partition_data_cifar100(
                            args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_num_in_total, args.batch_size,
                            args.num_classes_first_task, args.task_num, args.save_distribution_file,
                            args.one_task)

    dataset = [train_data_local_num_dict, train_data_local_dict, train_data_task_dict, class_num, task_classes]

    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    logging.info("class_num = " + str(output_dim))
    if model_name == "resnet_18":
        return resnet_18(class_num=output_dim)
    if model_name == "mobilenet":
        return mobilenet(alpha=args.width_multiplier, class_num=output_dim)


def custom_model_trainer(m_args, m_model, m_device):
    return MyModelTrainerCLS(m_args, m_model, m_device)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    # Set wandb writer
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        entity=args.wandb_entity,
        name=f"{args.model}_cnt{args.client_num_in_total}_cnr{args.client_num_per_round}_r{args.comm_rounds}" \
               f"_e{args.epochs}_lr{args.lr}_bs{args.batch_size}_{args.name}",
        project=args.dataset,
        group="afcl",
        config=vars(args),
        resume="allow"
    )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_training_data(args, args.dataset)

    # fractal pretraining
    pre_model = None
    if args.centralized_fractal_pretrain_steps or args.fractal_pretrain_rounds:
        pre_model = create_model(args, model_name=args.model, output_dim=1000)
        pre_model_trainer = custom_model_trainer(args, pre_model, device)
        fedavgAPI = FedAvgAPI(dataset, device, args, pre_model_trainer)
        if args.centralized_fractal_pretrain_steps:
            pre_model = fedavgAPI.centralized_fractal_pretraining()
        else:
            fedavgAPI.federated_fractal_pretraining()

    # create model
    model = create_model(args, model_name=args.model, output_dim=args.num_classes_first_task * 4)
    if args.fractal_pretrain_rounds or args.centralized_fractal_pretrain_steps:

        weight = pre_model.fc.weight.data
        bias = pre_model.fc.bias.data

        in_features = model.fc.in_features
        out_features = model.fc.out_features

        model.fc.weight.data = weight[:out_features]
        model.fc.bias.data = bias[:out_features]

    model_trainer = custom_model_trainer(args, model, device)
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)

    fedavgAPI.train_protos_global_local()
