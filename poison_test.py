############################################################
#
# poison_test.py
# Load poison examples from file and test
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################
import argparse
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms

from learning_module import now, get_model, load_model_from_checkpoint, get_transform
from learning_module import (
    train,
    test,
    adjust_learning_rate,
    to_log_file,
    to_results_table,
    PoisonedDataset,
    compute_perturbation_norms,
)


def main(args):
    """Main function to check the success rate of the given poisons
    input:
        args:       Argparse object
    return:
        void
    """
    print(now(), "poison_test.py main() running.")

    test_log = "poison_test_log.txt"
    to_log_file(args, args.output, test_log)

    lr = args.lr

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ####################################################
    #               Dataset

    # load the poisons and their indices within the training set from pickled files
    with open(os.path.join(args.poisons_path, "poisons.pickle"), "rb") as handle:
        poison_tuples = pickle.load(handle)
        print(len(poison_tuples), " poisons in this trial.")
        poisoned_label = poison_tuples[0][1]
    with open(os.path.join(args.poisons_path, "base_indices.pickle"), "rb") as handle:
        poison_indices = pickle.load(handle)

    # get datasets from torchvision
    if args.dataset.lower() == "cifar10":
        transform_train = get_transform(args.normalize, args.train_augment)
        transform_test = get_transform(args.normalize, False)
        cleanset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        num_classes = 10
    elif args.dataset.lower() == "tinyimagenet":
        print("tinyimagenet")
        transform_train = get_transform(args.normalize, args.train_augment, dataset=args.dataset)
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
        cleanset = torchvision.datasets.ImageFolder(
            "./data/tiny-imagenet-200/train", transform_train
        )
        testset = torchvision.datasets.ImageFolder(
            "./data/tiny-imagenet-200/test", transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, num_workers=1, shuffle=False
        )
        dataset = torchvision.datasets.ImageFolder(
            "./data/tiny-imagenet-200/train", transforms.ToTensor()
        )
        num_classes = 200
    else:
        print("Dataset not yet implemented. Exiting from poison_test.py.")
        sys.exit()

    trainset = PoisonedDataset(
        cleanset, poison_tuples, args.trainset_size, transform_train, poison_indices
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True
    )

    # get the target image from pickled file
    with open(os.path.join(args.poisons_path, "target.pickle"), "rb") as handle:
        target_img_tuple = pickle.load(handle)
        target_class = target_img_tuple[1]
        if len(target_img_tuple) == 4:
            patch = torch.tensor(target_img_tuple[2])
            if patch.shape[0] != 3 or patch.shape[1] != 5 or patch.shape[2] != 5:
                print(
                    "Expected shape of the patch is [3, 5, 5] but is {}. Exiting from poison_test.py.".format(
                        patch.shape
                    )
                )
                sys.exit()
            startx, starty = target_img_tuple[3]
            target_img_pil = target_img_tuple[0]
            h, w = target_img_pil.size
            if starty + 5 > h or startx + 5 > w:
                print(
                    "Invalid startx or starty point for the patch. Exiting from poison_test.py."
                )
                sys.exit()

            target_img_tensor = transforms.ToTensor()(target_img_pil)
            target_img_tensor[:, starty : starty + 5, startx : startx + 5] = patch
            target_img_pil = transforms.ToPILImage()(target_img_tensor)
        else:
            target_img_pil = target_img_tuple[0]

        target_img = transform_test(target_img_pil)

    poison_perturbation_norms = compute_perturbation_norms(
        poison_tuples, dataset, poison_indices
    )

    # the limit is '8/255' but we assert that it is smaller than 9/255 to account for PIL truncation.
    assert max(poison_perturbation_norms) - 9 / 255 < 1e-5, "Attack not clean label!"
    ####################################################

    ####################################################
    #           Network and Optimizer

    # load model from path if a path is provided
    if args.model_path is not None:
        net = load_model_from_checkpoint(
            args.model, args.model_path, args.pretrain_dataset
        )
    else:
        args.end2end = True  # we wouldn't fine tune from a random intiialization
        net = get_model(args.model, args.dataset)

    # freeze weights in feature extractor if not doing end2end retraining
    if not args.end2end:
        for param in net.parameters():
            param.requires_grad = False

    # reinitialize the linear layer
    num_ftrs = net.linear.in_features
    net.linear = nn.Linear(num_ftrs, num_classes)  # requires grad by default

    # set optimizer
    if args.optimizer.upper() == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=lr, weight_decay=args.weight_decay, momentum=0.9
        )
    elif args.optimizer.upper() == "ADAM":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    ####################################################

    ####################################################
    #        Poison and Train and Test
    print("==> Training network...")
    epoch = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_schedule, args.lr_factor)
        loss, acc = train(
            net, trainloader, optimizer, criterion, device, train_bn=args.end2end
        )

        if (epoch + 1) % args.val_period == 0:
            natural_acc = test(net, testloader, device)
            net.eval()
            p_acc = (
                net(target_img.unsqueeze(0).to(device)).max(1)[1].item()
                == poisoned_label
            )
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
                ", poison success: ",
                p_acc,
            )

    # test
    natural_acc = test(net, testloader, device)
    print(
        now(), " Training ended at epoch ", epoch, ", Natural accuracy: ", natural_acc
    )
    net.eval()
    p_acc = net(target_img.unsqueeze(0).to(device)).max(1)[1].item() == poisoned_label

    print(
        now(),
        " poison success: ",
        p_acc,
        " poisoned_label: ",
        poisoned_label,
        " prediction: ",
        net(target_img.unsqueeze(0).to(device)).max(1)[1].item(),
    )

    # Dictionary to write contest the csv file
    stats = OrderedDict(
        [
            ("poisons path", args.poisons_path),
            ("model", args.model_path if args.model_path is not None else args.model),
            ("target class", target_class),
            ("base class", poisoned_label),
            ("num poisons", len(poison_tuples)),
            ("max perturbation norm", np.max(poison_perturbation_norms)),
            ("epoch", epoch),
            ("loss", loss),
            ("training_acc", acc),
            ("natural_acc", natural_acc),
            ("poison_acc", p_acc),
        ]
    )
    to_results_table(stats, args.output)
    ####################################################

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--lr_schedule", nargs="+", default=[30], type=int, help="where to decrease lr"
    )
    parser.add_argument(
        "--lr_factor", default=0.1, type=float, help="factor by which to decrease lr"
    )
    parser.add_argument(
        "--epochs", default=40, type=int, help="number of epochs for training"
    )
    parser.add_argument(
        "--model", default="ResNet18", type=str, help="model for training"
    )
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
    parser.add_argument(
        "--pretrain_dataset",
        default="CIFAR100",
        type=str,
        help="dataset for pretrained network",
    )
    parser.add_argument("--optimizer", default="SGD", type=str, help="optimizer")
    parser.add_argument(
        "--val_period", default=20, type=int, help="print every __ epoch"
    )
    parser.add_argument(
        "--output", default="output_default", type=str, help="output subdirectory"
    )
    parser.add_argument(
        "--poisons_path",
        default="poison_examples",
        type=str,
        help="where are the poisons?",
    )
    parser.add_argument(
        "--model_path", default=None, type=str, help="where is the model saved?"
    )
    parser.add_argument(
        "--end2end", action="store_true", help="End to end retrain with poisons?"
    )
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--train_augment", dest="train_augment", action="store_true")
    parser.add_argument(
        "--no-train_augment", dest="train_augment", action="store_false"
    )
    parser.set_defaults(train_augment=False)
    parser.add_argument(
        "--weight_decay", default=0.0002, type=float, help="weight decay coefficient"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="training batch size"
    )
    parser.add_argument("--trainset_size", default=None, type=int, help="Trainset size")

    args = parser.parse_args()

    main(args)
