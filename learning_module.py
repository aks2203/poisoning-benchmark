############################################################
#
# learning_module.py
# Python module for deep learning
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################

import datetime
import os
import sys
import torch.utils.data as data
from models import *
import torch
import torchvision.transforms as transforms
import csv
import numpy as np

data_mean_std_dict = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet_all": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    "tinyimagenet_first": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    "tinyimagenet_last": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
}


def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


class PoisonedDataset(data.Dataset):
    def __init__(
        self, trainset, poison_instances, size=None, transform=None, poison_indices=None
    ):
        """ poison instances should be a list of tuples of poison examples
        and their respective labels like
            [(x_0, y_0), (x_1, y_1) ...]
        """
        super(PoisonedDataset, self).__init__()
        self.trainset = trainset
        self.poison_instances = poison_instances
        self.poison_indices = np.array([]) if poison_indices is None else poison_indices
        self.transform = transform
        self.dataset_size = size if size is not None else len(trainset)
        self.poisoned_label = (
            None if len(poison_instances) == 0 else poison_instances[0][1]
        )
        self.find_indices()

    def __getitem__(self, index):
        num_clean_samples = self.dataset_size - len(self.poison_instances)
        if index > num_clean_samples - 1:
            img, label = self.poison_instances[index - num_clean_samples]
            if self.transform is not None:
                img = self.transform(img)
            return img, label, 1  # last output is 1 for poison
        else:
            new_index = self.clean_indices[index]
            img, label = self.trainset[new_index]
            return img, label, 0  # last output is 0 for clean

    def __len__(self):
        return self.dataset_size

    def find_indices(self):
        good_idx = np.array([])
        batch_tar = np.array(self.trainset.targets)
        num_classes = len(set(batch_tar))
        num_per_class = int(self.dataset_size / num_classes)
        for label in range(num_classes):
            all_idx_for_this_class = np.where(batch_tar == label)[0]
            all_idx_for_this_class = np.setdiff1d(
                all_idx_for_this_class, self.poison_indices
            )
            this_class_idx = all_idx_for_this_class[:num_per_class]
            if label == self.poisoned_label and len(self.poison_instances) > 0:
                num_clean = num_per_class - len(self.poison_instances)
                this_class_idx = this_class_idx[:num_clean]
            good_idx = np.concatenate((good_idx, this_class_idx))

        self.clean_indices = good_idx.astype(int)


class NormalizeByChannelMeanStd(nn.Module):
    """Normalizing the input to the network
    """

    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return tensor.sub(mean).div(std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)


def to_log_file(out_dict, out_dir, log_name="log.txt"):
    """Function to write the logfiles
    input:
        out_dict:   Dictionary of content to be logged
        out_dir:    Path to store the log file
        log_name:   Name of the log file
    return:
        void
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, "a") as f:
        f.write(str(now()) + " " + str(out_dict) + "\n")


def to_results_table(stats, out_dir, log_name="results.csv"):
    """Function to write results in a csv file
    input:
        stats:      Dictionary of the content with keys as the column header
                    and values as the column value
        out_dir:    Path to store the csv file
        log_name:   Name of the csv file
    return:
        void
    """

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)
    try:
        with open(fname, "r") as f:
            pass
    except:
        with open(fname, "w") as f:
            fieldnames = list(stats.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(fname, "a") as f:
        fieldnames = list(stats.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(stats)


def adjust_learning_rate(optimizer, epoch, lr_schedule, lr_factor):
    """Function to decay the learning rate
    input:
        optimizer:      Pytorch optimizer object
        epoch:          Current epoch number
        lr_schedule:    Learning rate decay schedule list
        lr_factor:      Learning rate decay factor
    return:
        void
    """
    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_factor
        print(
            "Adjusting learning rate ",
            param_group["lr"] / lr_factor,
            "->",
            param_group["lr"],
        )
    return


def test(net, testloader, device):
    """Function to evaluate the performance of the model
    input:
        net:        Pytorch network object
        testloader: Pytorch dataloader object
        device:     Device on which data is to be loaded (cpu or gpu)
    return
        Testing accuracy
    """
    net.eval()
    natural_correct = 0
    total = 0
    results = {}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device)
            natural_outputs = net(inputs)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()

            total += targets.size(0)

    natural_acc = 100.0 * natural_correct / total
    results["Clean acc"] = natural_acc

    return natural_acc


def train(net, trainloader, optimizer, criterion, device, train_bn=True):
    """ Function to perform one epoch of training
    input:
        net:            Pytorch network object
        trainloader:    Pytorch dataloader object
        optimizer:      Pytorch optimizer object
        criterion:      Loss function

    output:
        train_loss:     Float, average loss value
        acc:            Float, percentage of training data correctly labeled
    """

    # Set net to train and zeros stats
    if train_bn:
        net.train()
    else:
        net.eval()

    net = net.to(device)

    train_loss = 0
    correct = 0
    total = 0
    poisons_correct = 0
    poisons_seen = 0
    for batch_idx, (inputs, targets, p) in enumerate(trainloader):
        inputs, targets, p = inputs.to(device), targets.to(device), p.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        poisons_correct += (predicted.eq(targets) * p).sum().item()
        poisons_seen += p.sum().item()
    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    return train_loss, acc


def get_transform(normalize, augment, dataset="CIFAR10"):
    """Function to perform required transformation on the tensor
    input:
        normalize:      Bool value to determine whether to normalize data
        augment:        Bool value to determine whether to augment data
        dataset:        Name of the dataset
    return
        Pytorch tranforms.Compose with list of all transformations
    """

    dataset = dataset.lower()
    mean, std = data_mean_std_dict[dataset]
    if "tinyimagenet" in dataset:
        dataset = "tinyimagenet"
    cropsize = {"cifar10": 32, "cifar100": 32, "tinyimagenet": 64}[dataset]
    padding = 4

    if normalize and augment:
        transform_list = [
            transforms.RandomCrop(cropsize, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    elif augment:
        transform_list = [
            transforms.RandomCrop(cropsize, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    elif normalize:
        transform_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    else:
        transform_list = [transforms.ToTensor()]

    return transforms.Compose(transform_list)


def get_model(model, dataset="CIFAR10"):
    """Function to load the model object
    input:
        model:      Name of the model
        dataset:    Name of the dataset
    return:
        net:        Pytorch Network Object
    """
    dataset = dataset.lower()
    model = model.lower()
    if dataset == "cifar10":
        if model == "resnet18":
            net = resnet18()
        elif model == "resnet32":
            net = resnet32()
        elif model == "mobilenetv2":
            net = MobileNetV2()
        elif model == "alexnet":
            net = AlexNet()
        elif model == "htbd_alexnet":
            net = HTBDAlexNet()
        elif model == "vgg11":
            net = vgg11()
        else:
            print(
                "Model not yet implemented. Exiting from learning_module.get_model()."
            )
            sys.exit()

    elif dataset == "cifar100":
        if model == "resnet18":
            net = resnet18(num_classes=100)
        elif model == "resnet32":
            net = resnet32(num_classes=100)
        elif model == "mobilenetv2":
            net = MobileNetV2(num_classes=100)
        elif model == "vgg11":
            net = vgg11(num_classes=100)
        else:
            print(
                "Model not yet implemented. Exiting from learning_module.get_model()."
            )
            sys.exit()

    elif dataset == "tinyimagenet_all":
        if model == "resnet34":
            net = resnet34(num_classes=200, conv1_size=7)
        elif model == "vgg16":
            net = vgg16(num_classes=200)
        elif model == "mobilenet_v2":
            net = MobileNetV2(num_classes=200)
    elif dataset == "tinyimagenet_first":
        if model == "resnet34":
            net = resnet34(num_classes=100, conv1_size=7)
        elif model == "vgg16":
            net = vgg16(num_classes=100)
        elif model == "mobilenet_v2":
            net = MobileNetV2(num_classes=100)
    elif dataset == "tinyimagenet_last":
        if model == "resnet34":
            net = resnet34(num_classes=100, conv1_size=7)
        elif model == "vgg16":
            net = vgg16(num_classes=100)
        elif model == "mobilenet_v2":
            net = MobileNetV2(num_classes=100)
    else:
        print("Dataset not yet implemented. Exiting from learning_module.get_model().")
        sys.exit()
    return net


def load_model_from_checkpoint(model, model_path, dataset="CIFAR10"):
    """Function to load the model from the given checkpoint
    input:
        model:          Name of the model to be loaded
        model_path:     Path of the checkpoint
        dataset:        Name of the dataset
    return:
        Pytorch Network Object
    """
    net = get_model(model, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict["net"])
    net = net.to(device)
    return net


def un_normalize_data(x, dataset="cifar10"):
    """Function to de-normalise image data
    input:
        x:      Tensor to be de-normalised
    return:
        De-normalised tensor
    """
    dataset = dataset.lower()
    mean, std = data_mean_std_dict[dataset]
    inv_mean = [-mean[i] / std[i] for i in range(len(mean))]
    inv_std = [1.0 / std[i] for i in range(len(std))]
    transform = transforms.Compose([transforms.Normalize(inv_mean, inv_std)])
    return transform(x)


def normalize_data(x, dataset="cifar10"):
    """Function to normalise image data
    input:
        x:      Tensor to be normalised
    return:
        Normalised tensor
    """
    dataset = dataset.lower()
    mean, std = data_mean_std_dict[dataset]
    transform = transforms.Compose([transforms.Normalize(mean, std)])
    return transform(x)


def compute_perturbation_norms(poisons, dataset, base_indices):
    """Function to compute the L-inf norm between poisons and original images
    input:
        poisons:        Tuple with poisoned images and labels
        dataset:        The whole dataset
        base_indices:   List of indices of the base images
    return:
        Array of L-inf norm between the poison and the base image
    """
    perturbation_norms = []
    poison_tensors = [transforms.ToTensor()(img) for img, label in poisons]
    for i, idx in enumerate(base_indices):
        perturbation_norms.append(
            torch.max(torch.abs(poison_tensors[i] - dataset[idx][0])).item()
        )
    return np.array(perturbation_norms)
