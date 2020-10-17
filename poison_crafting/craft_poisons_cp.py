############################################################
#
# craft_poisons_cp.py
# Convex Polytope Attack
# June 2020
#
# Reference: C. Zhu, W. R. Huang, H. Li, G. Taylor, C. Studer,
#     and T. Goldstein. Transferable clean-label poisoning attacks
#     on deep neural nets. In International Conference on Machine Learning,
#     pages 7614-7623, 2019.
############################################################
import argparse
import os
import pickle
import sys

sys.path.append(os.path.realpath("."))

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from ConvexPolytope.trainer import make_convex_polytope_poisons

from learning_module import (
    TINYIMAGENET_ROOT,
    now,
    data_mean_std_dict,
    get_transform,
    to_log_file,
    un_normalize_data,
    load_model_from_checkpoint,
)
from models import *
from tinyimagenet_module import TinyImageNet


def main(args):
    """Main function to generate the CP poisons
    inputs:
        args:           Argparse object
    reutrn:
        void
    """
    print(now(), "craft_poisons_cp.py main() running.")

    craft_log = "craft_log.txt"
    to_log_file(args, args.output, craft_log)

    ####################################################
    #               Dataset
    if args.dataset.lower() == "cifar10":
        transform_test = get_transform(args.normalize, False)
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_test
        )
    elif args.dataset.lower() == "tinyimagenet_first":
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
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
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
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
        transform_test = get_transform(args.normalize, False, dataset=args.dataset)
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
        print("Dataset not yet implemented. Exiting from craft_poisons_cp.py.")
        sys.exit()
    ###################################################

    ####################################################
    #         Find target and base images
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
    base_imgs = torch.stack([trainset[i][0] for i in base_indices])
    base_labels = [trainset[i][1] for i in base_indices]
    poisoned_label = base_labels[0]

    # log target and base details
    to_log_file("base indices: " + str(base_indices), args.output, craft_log)
    to_log_file("base labels: " + str(base_labels), args.output, craft_log)
    to_log_file("target_label: " + str(target_label), args.output, craft_log)
    to_log_file("target_index: " + str(target_img_idx), args.output, craft_log)

    # Set visible CUDA devices
    cudnn.benchmark = True

    # load the pre-trained models
    sub_net_list = []
    for n_model, chk_name in enumerate(args.model_path):
        net = load_model_from_checkpoint(
            args.model[n_model], chk_name, args.pretrain_dataset
        )
        sub_net_list.append(net)

    target_net = load_model_from_checkpoint(
        args.target_model, args.target_model_path, args.pretrain_dataset
    )

    # Get the target image
    target = target_img.unsqueeze(0)

    chk_path = args.poisons_path
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)

    base_tensor_list = [base_imgs[i] for i in range(base_imgs.shape[0])]
    base_tensor_list = [bt.to("cuda") for bt in base_tensor_list]

    poison_init = base_tensor_list
    mean, std = data_mean_std_dict[args.dataset.lower()]
    poison_tuple_list, recon_loss = make_convex_polytope_poisons(
        sub_net_list,
        target_net,
        base_tensor_list,
        target,
        device="cuda",
        opt_method=args.poison_opt,
        lr=args.poison_lr,
        momentum=args.poison_momentum,
        iterations=args.crafting_iters,
        epsilon=args.epsilon,
        decay_ites=args.poison_decay_ites,
        decay_ratio=args.poison_decay_ratio,
        mean=torch.Tensor(mean).reshape(1, 3, 1, 1),
        std=torch.Tensor(std).reshape(1, 3, 1, 1),
        chk_path=chk_path,
        poison_idxes=base_indices,
        poison_label=poisoned_label,
        tol=args.tol,
        start_ite=0,
        poison_init=poison_init,
        end2end=args.end2end,
    )

    # move poisons to PIL format
    if args.normalize:
        target = un_normalize_data(target.squeeze(0), args.dataset)
        for i in range(len(poison_tuple_list)):
            poison_tuple_list[i] = (
                transforms.ToPILImage()(
                    un_normalize_data(poison_tuple_list[i][0], args.dataset)
                ),
                poison_tuple_list[i][1],
            )
    else:
        target = target.squeeze(0)
        for i in range(len(poison_tuple_list)):
            poison_tuple_list[i] = (
                transforms.ToPILImage()(poison_tuple_list[i][0]),
                poison_tuple_list[i][1],
            )

    # get perturbation norms
    poison_perturbation_norms = []
    for idx, (poison_tensor, p_label) in enumerate(poison_tuple_list):
        poison_perturbation_norms.append(
            torch.max(
                torch.abs(
                    transforms.ToTensor()(poison_tensor)
                    - un_normalize_data(base_tensor_list[idx].cpu(), args.dataset)
                )
            ).item()
        )
    to_log_file(
        "perturbation norms: " + str(poison_perturbation_norms),
        args.output,
        "craft_log.txt",
    )

    ####################################################
    #        Save Poisons
    print(now(), "Saving poisons...")
    if not os.path.isdir(args.poisons_path):
        os.makedirs(args.poisons_path)
    with open(os.path.join(args.poisons_path, "poisons.pickle"), "wb") as handle:
        pickle.dump(poison_tuple_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        os.path.join(args.poisons_path, "perturbation_norms.pickle"), "wb"
    ) as handle:
        pickle.dump(poison_perturbation_norms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.poisons_path, "base_indices.pickle"), "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.poisons_path, "target.pickle"), "wb") as handle:
        pickle.dump(
            (transforms.ToPILImage()(target), target_label),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    to_log_file("poisons saved.", args.output, "craft_log.txt")
    ####################################################

    print(now(), "craft_poisons_cp.py done.")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convex Polytope Poison Attack")
    parser.add_argument(
        "--end2end",
        default=False,
        choices=[True, False],
        type=bool,
        help="Whether to consider an end-to-end victim",
    )
    parser.add_argument("--model", default=["ResNet18"], nargs="+", required=False)
    parser.add_argument(
        "--model_path",
        default=["pretrained_models/ResNet18_CIFAR100_A.pth"],
        nargs="+",
        type=str,
    )
    parser.add_argument("--target_model", default="resnet18", type=str)
    parser.add_argument(
        "--target_model_path",
        default="pretrained_models/ResNet18_CIFAR100_A.pth",
        type=str,
    )
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument(
        "--poisons_path",
        default="poison_examples/cp_poisons",
        type=str,
        help="Where to save the poisons?",
    )
    parser.add_argument(
        "--output", default="output_default", type=str, help="output directory"
    )
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
    parser.add_argument(
        "--pretrain_dataset",
        default="CIFAR100",
        type=str,
        help="dataset for pretrained network",
    )
    parser.add_argument(
        "--poison-lr",
        "-plr",
        default=4e-2,
        type=float,
        help="learning rate for making poison",
    )
    parser.add_argument(
        "--poison-momentum",
        "-pm",
        default=0.9,
        type=float,
        help="momentum for making poison",
    )
    parser.add_argument(
        "--crafting_iters", default=1200, type=int, help="iterations for making poison"
    )
    parser.add_argument(
        "--poison-decay-ites", type=int, metavar="int", nargs="+", default=[]
    )
    parser.add_argument("--poison-decay-ratio", default=0.1, type=float)
    parser.add_argument(
        "--epsilon",
        default=8 / 255,
        type=float,
        help="maximum deviation for each pixel",
    )
    parser.add_argument("--poison-opt", default="adam", type=str)
    parser.add_argument("--tol", default=1e-6, type=float)
    parser.add_argument(
        "--poison_setups",
        type=str,
        default="./poison_setups/cifar10_transfer_learning.pickle",
        help="poison setup pickle file",
    )
    parser.add_argument("--setup_idx", type=int, default=0, help="Which setup to use")
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

    if args.target_model_path is None:
        args.target_model_path = args.model_path

    main(args)
