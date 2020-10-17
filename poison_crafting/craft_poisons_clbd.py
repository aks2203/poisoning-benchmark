############################################################
#
# craft_poisons_clbd.py
# Clean-label Backdoor Attack
# June 2020
#
# Reference: A. Turner, D. Tsipras, and A. Madry. Clean-label
#     backdoor attacks. 2018.
############################################################
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image
from torchvision import transforms

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


class AttackPGD(nn.Module):
    """Class for the PGD adversarial attack"""

    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = config["step_size"]
        self.epsilon = config["epsilon"]
        self.num_steps = config["num_steps"]

    def forward(self, inputs, targets):
        """Forward function for the nn class
        inputs:
            inputs:     The input to the network
            targets:    True labels
        reutrn:
            adversarially perturbed inputs
        """
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = nn.functional.cross_entropy(
                    self.basic_net(x), targets, reduction="sum"
                )
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return x


def main(args):
    """Main function to generate the CLBD poisons
    inputs:
        args:           Argparse object
    reutrn:
        void
    """
    print(now(), "craft_poisons_clbd.py main() running...")
    mean, std = data_mean_std_dict[args.dataset.lower()]
    mean = list(mean)
    std = list(std)
    normalize_net = NormalizeByChannelMeanStd(mean, std)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_checkpoint(
        args.model[0], args.model_path[0], args.pretrain_dataset
    )
    model.eval()
    if args.normalize:
        model = nn.Sequential(normalize_net, model)
    model = model.to(device)

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
    else:
        print("Dataset not yet implemented. Exiting from craft_poisons_clbd.py.")
        sys.exit()
    ###################################################

    with open(args.poison_setups, "rb") as handle:
        setup_dicts = pickle.load(handle)
    setup = setup_dicts[args.setup_idx]

    target_img_idx = (
        setup["target index"] if args.target_img_idx is None else args.target_img_idx
    )
    base_indices = (
        setup["base indices"] if args.base_indices is None else args.base_indices
    )

    # get single target
    target_img, target_label = testset[target_img_idx]

    # get multiple bases
    base_imgs = torch.stack([trainset[i][0] for i in base_indices]).to(device)
    base_labels = torch.LongTensor([trainset[i][1] for i in base_indices]).to(device)

    # get attacker
    config = {
        "epsilon": args.epsilon,
        "step_size": args.step_size,
        "num_steps": args.num_steps,
    }
    attacker = AttackPGD(model, config)

    # get patch
    trans_trigger = transforms.Compose(
        [transforms.Resize((args.patch_size, args.patch_size)), transforms.ToTensor()]
    )
    trigger = Image.open("./poison_crafting/triggers/clbd.png").convert("RGB")
    trigger = trans_trigger(trigger).unsqueeze(0).to(device)

    # craft poisons
    num_batches = int(np.ceil(base_imgs.shape[0] / 1000))
    batches = [
        (base_imgs[1000 * i : 1000 * (i + 1)], base_labels[1000 * i : 1000 * (i + 1)])
        for i in range(num_batches)
    ]

    # attack all the bases
    adv_batches = []
    for batch_img, batch_labels in batches:
        adv_batches.append(attacker(batch_img, batch_labels))
    adv_bases = torch.cat(adv_batches)

    # Starting coordinates of the patch
    start_x = args.image_size - args.patch_size
    start_y = args.image_size - args.patch_size

    # Mask
    mask = torch.ones_like(adv_bases)

    # uncomment for patching all corners
    mask[
        :, start_y : start_y + args.patch_size, start_x : start_x + args.patch_size
    ] = 0
    # mask[:, 0 : args.patch_size, start_x : start_x + args.patch_size] = 0
    # mask[:, start_y : start_y + args.patch_size, 0 : args.patch_size] = 0
    # mask[:, 0 : args.patch_size, 0 : args.patch_size] = 0

    pert = (adv_bases - base_imgs) * mask
    adv_bases_masked = base_imgs + pert

    # Attching patch to the masks
    for i in range(len(base_imgs)):
        # uncomment for patching all corners
        adv_bases_masked[
            i,
            :,
            start_y : start_y + args.patch_size,
            start_x : start_x + args.patch_size,
        ] = trigger
        # adv_bases_masked[
        #     i, :, 0 : args.patch_size, start_x : start_x + args.patch_size
        # ] = trigger
        # adv_bases_masked[
        #     i, :, start_y : start_y + args.patch_size, 0 : args.patch_size
        # ] = torch.flip(trigger, (-1,))
        # adv_bases_masked[i, :, 0 : args.patch_size, 0 : args.patch_size] = torch.flip(
        #     trigger, (-1,)
        # )

    final_pert = torch.clamp(adv_bases_masked - base_imgs, -args.epsilon, args.epsilon)
    poisons = base_imgs + final_pert

    poisons = poisons.clamp(0, 1).cpu()
    poisoned_tuples = [
        (transforms.ToPILImage()(poisons[i]), base_labels[i].item())
        for i in range(poisons.shape[0])
    ]

    target_tuple = (
        transforms.ToPILImage()(target_img),
        int(target_label),
        trigger.squeeze(0).cpu(),
        [start_x, start_y],
    )

    ####################################################
    #        Save Poisons
    print(now(), "Saving poisons...")
    if not os.path.isdir(args.poisons_path):
        os.makedirs(args.poisons_path)
    with open(os.path.join(args.poisons_path, "poisons.pickle"), "wb") as handle:
        pickle.dump(poisoned_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.poisons_path, "target.pickle"), "wb") as handle:
        pickle.dump(
            target_tuple,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with open(os.path.join(args.poisons_path, "base_indices.pickle"), "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ####################################################

    print(now(), "craft_poisons_clbd.py done.")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean-label Backdoor Attack")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset")
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=False)
    parser.add_argument(
        "--epsilon", type=int, default=8 / 255, help="poison perturbation allowance"
    )
    parser.add_argument(
        "--model", type=str, default=["resnet18"], nargs="+", help="model name"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=["pretrained_models/ResNet18_CIFAR10_adv.pth"],
        nargs="+",
    )
    parser.add_argument(
        "--pretrain_dataset", default="CIFAR10", type=str, help="dataset"
    )
    parser.add_argument("--image_size", type=int, default=32, help="Image Size")
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of PGD steps")
    parser.add_argument(
        "--step_size", type=int, default=2 / 255, help="Step size for perturbation"
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
        default="poison_examples/clbd_poisons",
        type=str,
        help="Where to save the poisons?",
    )
    parser.add_argument(
        "--target_img_idx",
        default=None,
        type=int,
        help="Index of the target image in the claen set.",
    )
    parser.add_argument(
        "--base_indices", nargs="+", default=None, type=int, help="which base images"
    )

    args = parser.parse_args()

    if args.dataset.lower() == "cifar10":
        args.image_size = 32
    elif "tinyimagenet" in args.dataset.lower():
        args.image_size = 64
        args.trigger_path = "poison_crafting/triggers/clbd_8.png"

    main(args)
