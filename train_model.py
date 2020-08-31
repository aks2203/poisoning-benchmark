#######################################################
#
# train_model.py
# Train and save models
# Developed as part of Poison Attack Benchmarking project
# June 2019
#
############################################################
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from learning_module import (
    train,
    test,
    adjust_learning_rate,
    to_log_file,
    now,
    get_model,
    PoisonedDataset,
    get_transform,
)


def main(args):
    """Main function to train and test a model
    input:
        args:       Argparse object that contains all the parsed values
    return:
        void
    """

    print(now(), "train_model.py main() running.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_log = "train_log.txt"
    to_log_file(args, args.output, train_log)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ####################################################
    #               Load the Dataset
    if args.dataset.lower() == "cifar10":
        transform_train = get_transform(args.normalize, args.train_augment)
        transform_test = get_transform(args.normalize, False)
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainset = PoisonedDataset(
            trainset, (), args.trainset_size, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )
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
        trainset = PoisonedDataset(
            trainset, (), args.trainset_size, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    elif args.dataset.lower() == "tinyimagenet":
        transform_train = get_transform(args.normalize, args.train_augment, dataset=args.dataset)
        transform_test = get_transform(args.normalize, args.test_augment, dataset=args.dataset)
        trainset = torchvision.datasets.ImageFolder("./data/tiny-imagenet-200/train", transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=1, shuffle=True)
        testset = torchvision.datasets.ImageFolder("./data/tiny-imagenet-200/test", transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=1, shuffle=False)

    else:
        print("Dataset not yet implemented. Ending run from train_model.py.")
        sys.exit()

    ####################################################

    ####################################################
    #           Network and Optimizer
    net = get_model(args.model, args.dataset)
    net = net.to(device)
    start_epoch = 0

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr, weight_decay=2e-4, momentum=0.9
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()

    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=device)
        net.load_state_dict(state_dict["net"])
        optimizer.load_state_dict(state_dict["optimizer"])
        start_epoch = state_dict["epoch"]
    ####################################################

    ####################################################
    #        Train and Test
    print("==> Training network...")
    loss = 0
    all_losses = []
    epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_schedule, args.lr_factor)
        loss, acc = train(net, trainloader, optimizer, criterion, device)
        all_losses.append(loss)

        if (epoch + 1) % args.val_period == 0:
            natural_acc = test(net, testloader, device)
            print(
                now(),
                " Epoch: ",
                epoch,
                ", Loss: ",
                loss,
                ", Training acc: ",
                acc,
                ", Natural accuracy: ",
                natural_acc,
            )
            to_log_file(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "training_acc": acc,
                    "natural_acc": natural_acc,
                },
                args.output,
                train_log,
            )

    # test
    natural_acc = test(net, testloader, device)
    print(
        now(), " Training ended at epoch ", epoch, ", Natural accuracy: ", natural_acc
    )
    to_log_file(
        {"epoch": epoch, "loss": loss, "natural_acc": natural_acc},
        args.output,
        train_log,
    )
    ####################################################

    ####################################################
    #        Save
    if args.save_net:
        state = {
            "net": net.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
        }
        out_str = os.path.join(
            args.checkpoint,
            args.model
            + "_seed_"
            + str(args.seed)
            + "_normalize="
            + str(args.normalize)
            + "_augment="
            + str(args.train_augment)
            + "_optimizer="
            + str(args.optimizer)
            + "_epoch="
            + str(epoch)
            + ".pth",
        )
        print("Saving model to: ", args.checkpoint, " out_str: ", out_str)
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)
        torch.save(state, out_str)
    ####################################################

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Poisoning Benchmark")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument(
        "--lr_schedule",
        nargs="+",
        default=[100, 150],
        type=int,
        help="when to decrease lr",
    )
    parser.add_argument(
        "--lr_factor", default=0.1, type=float, help="factor by which to decrease lr"
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of epochs for training"
    )
    parser.add_argument("--optimizer", default="SGD", type=str, help="optimizer")
    parser.add_argument(
        "--model", default="ResNet18", type=str, help="model for training"
    )
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
    parser.add_argument("--trainset_size", default=None, type=int, help="Trainset size")
    parser.add_argument(
        "--val_period", default=20, type=int, help="print every __ epoch"
    )
    parser.add_argument(
        "--output", default="output_default", type=str, help="output subdirectory"
    )
    parser.add_argument(
        "--checkpoint",
        default="check_default",
        type=str,
        help="where to save the network",
    )
    parser.add_argument(
        "--model_path", default=None, type=str, help="where is the model saved?"
    )
    parser.add_argument("--save_net", action="store_true", help="save net?")
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for seeding random processes."
    )
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--train_augment", dest="train_augment", action="store_true")
    parser.add_argument(
        "--no-train_augment", dest="train_augment", action="store_false"
    )
    parser.set_defaults(train_augment=False)
    args = parser.parse_args()

    main(args)
