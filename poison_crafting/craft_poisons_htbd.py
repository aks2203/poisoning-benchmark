############################################################
#
# craft_poisons_htbd.py
# Hidden Trigger Backdoor Attack
# June 2020
#
# Reference: A. Saha, A. Subramanya, and H. Pirsiavash. Hidden
#     trigger backdoor attacks. arXiv:1910.00033, 2019.
############################################################
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

sys.path.append(os.path.realpath("."))
from learning_module import (
    TINYIMAGENET_ROOT,
    load_model_from_checkpoint,
    now,
    get_transform,
    NormalizeByChannelMeanStd,
    data_mean_std_dict,
)
from tinyimagenet_module import TinyImageNet


class LossMeter(object):
    """Computes and stores the average and current loss value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, iteration, dataset):
    """Update the learning rate by a factor after certain iterations
    inputs:
        lr:         Current learning rate
        iter:       Current iteration number
        dataset:  Name of the dataset
    return:
        updated learning rate
    """
    if dataset.upper() == "CIFAR10":
        lr = lr * (0.95 ** (iteration // 2000))
        return lr
    else:
        lr = lr * (0.5 ** (iteration // 1000))
        return lr


def main(args):
    """Function to generate the generate the HTBD poison
    inputs:
        args:        Argument Parser object
    return:
        void
    """
    print(now(), "craft_poisons_htbd.py main() running...")
    mean, std = data_mean_std_dict[args.dataset.lower()]
    normalization_net = NormalizeByChannelMeanStd(mean, std)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model_from_checkpoint(
        args.model[0], args.model_path[0], args.pretrain_dataset
    )
    net.eval()
    normalization_net = normalization_net.to(device)
    net = net.to(device)

    ####################################################
    #               Dataset
    if args.dataset.lower() == "cifar10":
        transform_test = get_transform(False, False)
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_test
        )
        num_per_class = 5000
    elif args.dataset.lower() == "tinyimagenet_first":
        transform_test = get_transform(False, False, dataset=args.dataset)
        trainset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_test,
            classes="firsthalf",
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="firsthalf",
        )
        num_per_class = 500
    elif args.dataset.lower() == "tinyimagenet_last":
        transform_test = get_transform(False, False, dataset=args.dataset)
        trainset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_test,
            classes="lasthalf",
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="lasthalf",
        )
        num_per_class = 500
    elif args.dataset.lower() == "tinyimagenet_all":
        transform_test = get_transform(False, False, dataset=args.dataset)
        trainset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="train",
            transform=transform_test,
            classes="all",
        )
        testset = TinyImageNet(
            TINYIMAGENET_ROOT,
            split="val",
            transform=transform_test,
            classes="all",
        )
        num_per_class = 500
    else:
        print("Dataset not yet implemented. Exiting from craft_poisons_htbd.py.")
        sys.exit()
    ###################################################

    with open(args.poison_setups, "rb") as handle:
        setup_dicts = pickle.load(handle)
    setup = setup_dicts[args.setup_idx]

    losses = LossMeter()
    lr = args.lr

    trans_trigger = transforms.Compose(
        [transforms.Resize((args.patch_size, args.patch_size)), transforms.ToTensor()]
    )
    trigger = Image.open(args.trigger_path).convert("RGB")
    trigger = trans_trigger(trigger).unsqueeze(0).to(device)

    target_img_idx = (
        setup["target index"] if args.target_img_idx is None else args.target_img_idx
    )
    base_indices = (
        setup["base indices"] if args.base_indices is None else args.base_indices
    )

    target_img_save, target_label_save = testset[target_img_idx]
    target_class = target_label_save

    # Get target images
    trainset_targets = np.array(trainset.targets)
    target_class = target_class
    tar_idx = np.where(trainset_targets == target_class)[0]
    indexes = np.arange(0, num_per_class)
    indexes = np.random.choice(indexes, len(base_indices), replace=False)
    target_img_idx = np.array(tar_idx[indexes]).astype(int)

    poisoned_tuples = []
    target_tuples = []

    # get multiple bases
    base_imgs = torch.stack([trainset[i][0] for i in base_indices]).to(device)
    base_labels = torch.LongTensor([trainset[i][1] for i in base_indices]).to(device)

    target_imgs = torch.stack([trainset[i][0] for i in target_img_idx]).to(device)

    batch_size = base_imgs.shape[0]

    start_x = args.image_size - args.patch_size - 5
    start_y = args.image_size - args.patch_size - 5

    for i in range(0, len(base_imgs), batch_size):

        remaining = np.min((batch_size, len(base_imgs) - i))
        input_target_imgs = target_imgs[i : i + remaining]
        input_bases = base_imgs[i : i + remaining]
        input_target_imgs = input_target_imgs.to(device)
        input_bases = input_bases.to(device)

        for z in range(input_target_imgs.size(0)):
            # paste the trigger on input_target_imgs
            input_target_imgs[
                z,
                :,
                start_y : start_y + args.patch_size,
                start_x : start_x + args.patch_size,
            ] = trigger

        # get features of input_target_imgs
        if args.normalize:
            input_target_imgs = normalization_net(input_target_imgs)
        feat1 = net(input_target_imgs, penu=True)
        feat1 = feat1.detach().clone()

        for j in range(args.crafting_iters):
            input_bases.requires_grad_()
            lr1 = adjust_learning_rate(lr, j, args.dataset)

            # get features of input_bases
            if args.normalize:
                input_bases_proc = normalization_net(input_bases)
            else:
                input_bases_proc = input_bases

            feat2 = net(input_bases_proc, penu=True)
            feat11 = feat1.clone()
            dist = torch.cdist(feat1, feat2)
            for _ in range(feat2.size(0)):

                dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
                feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
                dist[dist_min_index[0], dist_min_index[1]] = 1e5

            loss = torch.norm(feat1 - feat2) ** 2
            losses.update(loss.item(), input_target_imgs.size(0))
            loss.backward()

            input_bases = input_bases - lr1 * input_bases.grad
            pert = input_bases - base_imgs[i : i + remaining]
            pert = torch.clamp(pert, -args.epsilon, args.epsilon).detach_()
            input_bases = pert + base_imgs[i : i + remaining]
            input_bases = input_bases.clamp(0, 1)

            if j % 100 == 0:
                logging.info(
                    "Epoch: {:2d} | i: {} | iter: {:5d} | LR: {:2.5f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}".format(
                        0, i, j, lr1, losses.val, losses.avg
                    )
                )
                print(
                    "Epoch: {:2d} | i: {} | iter: {:5d} | LR: {:2.5f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}".format(
                        0, i, j, lr1, losses.val, losses.avg
                    )
                )

            if loss.item() < 10 or j == (args.crafting_iters - 1):
                logging.info("Max_Loss: {}".format(loss.item()))
                for k in range(input_bases.size(0)):
                    poisoned_tuples.append(
                        (
                            transforms.ToPILImage()(input_bases[k].cpu()),
                            int(base_labels[i + k]),
                        )
                    )
                    if len(target_tuples) < 1:
                        img = target_img_save.to(device)
                        trigger_ = trigger.cpu()

                        # Tuple for target.pickle
                        target_tuples.append(
                            (
                                transforms.ToPILImage()(img.cpu()),
                                int(target_label_save),
                                trigger_.squeeze(0),
                                [start_x, start_y],
                            )
                        )
                break

    # Creating the directories and saving the poisons
    print(now(), "Saving poisons...")
    if not os.path.isdir(args.poisons_path):
        os.makedirs(args.poisons_path)
    with open(os.path.join(args.poisons_path, "poisons.pickle"), "wb") as handle:
        pickle.dump(poisoned_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.poisons_path, "target.pickle"), "wb") as handle:
        pickle.dump(target_tuples[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.poisons_path, "base_indices.pickle"), "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ####################################################

    print(now(), "craft_poisons_htbd.py done.")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hidden Trigger Backdoor Attack")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset")
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument(
        "--epsilon", type=int, default=8 / 255, help="poison perturbation allowance"
    )
    parser.add_argument(
        "--model", type=str, default=["resnet18"], nargs="+", help="model name"
    )
    parser.add_argument("--image_size", type=int, default=32, help="Image Size")
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
    parser.add_argument(
        "--trigger_path",
        type=str,
        default="./poison_crafting/triggers/htbd.png",
        help="Trigger path",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch Size")
    parser.add_argument(
        "--crafting_iters", type=int, default=5000, help="Number of iterations"
    )
    parser.add_argument(
        "--poison_setups",
        type=str,
        default="./poison_setups/cifar10_transfer_learning.pickle",
        help="poison setup pickle file",
    )
    parser.add_argument("--setup_idx", type=int, default=0, help="Which setup to use")
    parser.add_argument(
        "--poisons_path",
        default="poison_examples/htbd_poisons",
        type=str,
        help="Where to save the poisons?",
    )
    parser.add_argument(
        "--base_indices", nargs="+", default=None, type=int, help="which base images"
    )
    parser.add_argument(
        "--target_img_idx",
        default=None,
        type=int,
        help="Index of the target image in the clean set.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        nargs="+",
        default=["pretrained_models/ResNet18_CIFAR100_A.pth"],
        help="Checkpoint file",
    )
    parser.add_argument(
        "--pretrain_dataset", default="CIFAR100", type=str, help="dataset"
    )

    args = parser.parse_args()

    if args.dataset.lower() == "cifar10":
        args.image_size = 32
    elif "tinyimagenet" in args.dataset.lower():
        args.image_size = 64

    main(args)
