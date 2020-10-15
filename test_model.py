############################################################
#
# test_model.py
# Load trained model and test its natural performance
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################
import argparse
import sys
from collections import OrderedDict

import torch
import torch.utils.data as data
import torchvision

from learning_module import (
    TINYIMAGENET_ROOT,
    test,
    to_log_file,
    to_results_table,
    now,
    get_model,
    load_model_from_checkpoint,
    get_transform,
)
from tinyimagenet_module import TinyImageNet


def main(args):
    """Main function to test a model
    input:
        args:       Argparse object that contains all the parsed values
    return:
        void
    """

    print(now(), "test_model.py main() running.")

    test_log = "clean_test_log.txt"
    to_log_file(args, args.output, test_log)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ####################################################
    #               Dataset
    if args.dataset.lower() == "cifar10":
        transform_train = get_transform(args.normalize, args.train_augment)
        transform_test = get_transform(args.normalize, False)
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128)
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    elif args.dataset.lower() == "cifar100":
        transform_train = get_transform(args.normalize, args.train_augment)
        transform_test = get_transform(args.normalize, False)
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128)
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    elif args.dataset.lower() == "tinyimagenet_first":
        transform_train = get_transform(
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        trainset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_train,
            classes="firsthalf",
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, num_workers=1, shuffle=True
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="firsthalf",
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, num_workers=1, shuffle=False
        )

    elif args.dataset.lower() == "tinyimagenet_last":
        transform_train = get_transform(
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        trainset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_train,
            classes="lasthalf",
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, num_workers=1, shuffle=True
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="lasthalf",
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, num_workers=1, shuffle=False
        )

    elif args.dataset.lower() == "tinyimagenet_all":
        transform_train = get_transform(
            args.normalize, args.train_augment, dataset=args.dataset
        )
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        trainset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_train,
            classes="all",
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, num_workers=1, shuffle=True
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="all",
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, num_workers=1, shuffle=False
        )

    else:
        print("Dataset not yet implemented. Exiting from test_model.py.")
        sys.exit()

    ####################################################

    ####################################################
    #           Network and Optimizer
    net = get_model(args.model, args.dataset)

    # load model from path if a path is provided
    if args.model_path is not None:
        net = load_model_from_checkpoint(args.model, args.model_path, args.dataset)
    else:
        print("No model path provided, continuing test with untrained network.")
    net = net.to(device)
    ####################################################

    ####################################################
    #        Test Model
    training_acc = test(net, trainloader, device)
    natural_acc = test(net, testloader, device)
    print(now(), " Training accuracy: ", training_acc)
    print(now(), " Natural accuracy: ", natural_acc)
    stats = OrderedDict(
        [
            ("model path", args.model_path),
            ("model", args.model),
            ("normalize", args.normalize),
            ("augment", args.train_augment),
            ("training_acc", training_acc),
            ("natural_acc", natural_acc),
        ]
    )
    to_results_table(stats, args.output, "clean_performance.csv")
    ####################################################

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")
    parser.add_argument(
        "--model", default="ResNet18", type=str, help="model for training"
    )
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
    parser.add_argument(
        "--output", default="output_default", type=str, help="output subdirectory"
    )
    parser.add_argument(
        "--model_path", default=None, type=str, help="where is the model saved?"
    )
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--train_augment", dest="train_augment", action="store_true")
    args = parser.parse_args()

    main(args)
