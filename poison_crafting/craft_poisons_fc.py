############################################################
#
# craft_poisons_fc.py
# Feature Collision Attack
# June 2020
#
# Reference: A. Shafahi, W. R. Huang, M. Najibi, O. Suciu, C. Studer,
#     T. Dumitras, and T. Goldstein. Poison frogs! targeted clean-label
#     poisoning attacks on neural networks.
#     In Advances in Neural Information Processing Systems, pages 6103-6113, 2018.
############################################################
import argparse
import copy
import os
import pickle
import sys

import torch
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.realpath("."))
from learning_module import get_transform
from learning_module import (
    TINYIMAGENET_ROOT,
    to_log_file,
    now,
    normalize_data,
    un_normalize_data,
    load_model_from_checkpoint,
)
from tinyimagenet_module import TinyImageNet


def main(args):
    """Main function to generate the FC poisons
    inputs:
        args:           Argparse object
    reutrn:
        void
    """
    print(now(), "craft_poisons_fc.py main() running.")

    craft_log = "craft_log.txt"
    to_log_file(args, args.output, craft_log)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ####################################################
    #               Dataset
    if args.dataset.lower() == "cifar10":
        transform_test = get_transform(args.normalize, False)
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_test
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
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
        print("Dataset not yet implemented. Exiting from craft_poisons_fc.py.")
        sys.exit()
    ###################################################

    ####################################################
    #          Craft and insert poison image
    feature_extractors = []
    for i in range(len(args.model)):
        feature_extractors.append(
            load_model_from_checkpoint(
                args.model[i], args.model_path[i], args.pretrain_dataset
            )
        )

    for i in range(len(feature_extractors)):
        for param in feature_extractors[i].parameters():
            param.requires_grad = False
        feature_extractors[i].eval()
        feature_extractors[i] = feature_extractors[i].to(device)

    with open(args.poison_setups, "rb") as handle:
        setup_dicts = pickle.load(handle)
    setup = setup_dicts[args.setup_idx]

    target_img_idx = (
        setup["target index"] if args.target_img_idx is None else args.target_img_idx
    )
    base_indices = (
        setup["base indices"] if args.base_indices is None else args.base_indices
    )
    # Craft poisons
    poison_iterations = args.crafting_iters
    poison_perturbation_norms = []

    # get single target
    target_img, target_label = testset[target_img_idx]

    # get multiple bases
    base_imgs = torch.stack([trainset[i][0] for i in base_indices])
    base_labels = torch.LongTensor([trainset[i][1] for i in base_indices])

    # log target and base details
    to_log_file("base indices: " + str(base_indices), args.output, craft_log)
    to_log_file("base labels: " + str(base_labels), args.output, craft_log)
    to_log_file("target_label: " + str(target_label), args.output, craft_log)
    to_log_file("target_index: " + str(target_img_idx), args.output, craft_log)

    # fill list of tuples of poison images and labels
    poison_tuples = []
    target_img = (
        un_normalize_data(target_img, args.dataset) if args.normalize else target_img
    )
    beta = 4.0 if args.normalize else 0.1

    base_tuples = list(zip(base_imgs, base_labels))
    for base_img, label in base_tuples:
        # unnormalize the images for optimization
        b_unnormalized = (
            un_normalize_data(base_img, args.dataset) if args.normalize else base_img
        )
        objective_vals = [10e8]
        step_size = args.step_size

        # watermarking
        x = copy.deepcopy(b_unnormalized)
        x = args.watermark_coeff * target_img + (1 - args.watermark_coeff) * x

        # feature collision optimization
        done_with_fc = False
        i = 0
        while not done_with_fc and i < poison_iterations:
            x.requires_grad = True
            if args.normalize:
                mini_batch = torch.stack(
                    [
                        normalize_data(x, args.dataset),
                        normalize_data(target_img, args.dataset),
                    ]
                ).to(device)
            else:
                mini_batch = torch.stack([x, target_img]).to(device)

            loss = 0
            for feature_extractor in feature_extractors:
                feats = feature_extractor.penultimate(mini_batch)
                loss += torch.norm(feats[0, :] - feats[1, :]) ** 2
            grad = torch.autograd.grad(loss, [x])[0]
            x_hat = x.detach() - step_size * grad.detach()
            if not args.l2:
                pert = (x_hat - b_unnormalized).clamp(-args.epsilon, args.epsilon)
                x_new = b_unnormalized + pert
                x_new = x_new.clamp(0, 1)
                obj = loss

            else:
                x_new = (
                    x_hat.detach() + step_size * beta * b_unnormalized.detach()
                ) / (1 + step_size * beta)
                x_new = x_new.clamp(0, 1)
                obj = beta * torch.norm(x_new - b_unnormalized) ** 2 + loss

            if obj > objective_vals[-1]:
                step_size *= 0.2

            else:
                if torch.norm(x - x_new) / torch.norm(x) < 1e-5:
                    done_with_fc = True
                x = copy.deepcopy(x_new)
                objective_vals.append(obj)
            i += 1
        poison_tuples.append((transforms.ToPILImage()(x), label.item()))
        poison_perturbation_norms.append(
            torch.max(torch.abs(x - b_unnormalized)).item()
        )
        x.requires_grad = False
    ####################################################

    ####################################################
    #        Save Poisons
    print(now(), "Saving poisons...")
    if not os.path.isdir(args.poisons_path):
        os.makedirs(args.poisons_path)
    with open(os.path.join(args.poisons_path, "poisons.pickle"), "wb") as handle:
        pickle.dump(poison_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        os.path.join(args.poisons_path, "perturbation_norms.pickle"), "wb"
    ) as handle:
        pickle.dump(poison_perturbation_norms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.poisons_path, "base_indices.pickle"), "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.poisons_path, "target.pickle"), "wb") as handle:
        pickle.dump(
            (transforms.ToPILImage()(target_img), target_label),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    to_log_file("poisons saved.", args.output, craft_log)
    ####################################################

    print(now(), "craft_poisons_fc.py done.")
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature Collision Poison Attack")
    parser.add_argument(
        "--model", default=["resnet18"], nargs="+", type=str, help="model for training"
    )
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
    parser.add_argument(
        "--pretrain_dataset", default="CIFAR100", type=str, help="dataset"
    )
    parser.add_argument(
        "--model_path",
        default=["pretrained_models/ResNet18_CIFAR100_A.pth"],
        nargs="+",
        type=str,
        help="Model path",
    )
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument(
        "--output", default="output_default", type=str, help="output subdirectory"
    )
    parser.add_argument(
        "--crafting_iters",
        default=120,
        type=int,
        help="How many iterations when crafting poison?",
    )
    parser.add_argument(
        "--watermark_coeff", default=0.3, type=float, help="Opacity of watermark."
    )
    parser.add_argument(
        "--step_size", default=0.001, type=float, help="Step size in optimization"
    )
    parser.add_argument(
        "--l2", action="store_true", help="use l-2 constrained version of attack."
    )
    parser.add_argument(
        "--epsilon", default=8 / 255, type=float, help="poison perturbation allowance"
    )
    parser.add_argument(
        "--poisons_path",
        default="poison_examples/fc_poisons",
        type=str,
        help="Where to save the poisons?",
    )
    parser.add_argument(
        "--poison_setups",
        type=str,
        default="poison_setups_transfer_learning.pickle",
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

    main(args)
