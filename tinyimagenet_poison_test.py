############################################################
#
# benchmark_test.py
# Code to execute benchmark tests
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################
import argparse
import os

import poison_test
from learning_module import now

whitebox_modelpath = "/cmlscratch/avi1/poisoning-benchmark/tinyimagenet_firsthalf_checkpoints/vgg16_seed_1000_normalize=True_augment=True_optimizer=SGD_epoch=199.pth"
greybox_modelpath = "/cmlscratch/avi1/poisoning-benchmark/tinyimagenet_firsthalf_checkpoints/vgg16_seed_1001_normalize=True_augment=True_optimizer=SGD_epoch=199.pth"
blackbox_modelpath = [
    "/cmlscratch/avi1/poisoning-benchmark/tinyimagenet_firsthalf_checkpoints/resnet34_seed_1000_normalize=True_augment=True_optimizer=SGD_epoch=199.pth",
    "/cmlscratch/avi1/poisoning-benchmark/tinyimagenet_firsthalf_checkpoints/mobilenet_v2_seed_1000_normalize=True_augment=True_optimizer=SGD_epoch=199.pth"
]


def main(args):
    """Main function to run a benchmark test
    input:
        args:       Argparse object that contains all the parsed values
    return:
        void
    """

    print(now(), "benchmark_test.py running.")
    out_dir = args.output
    if not args.from_scratch:
        print(
            f"Testing poisons from {args.poisons_path}, in the transfer learning setting...\n".format()
        )

        args.dataset = "tinyimagenet_last"
        args.pretrain_dataset = "tinyimagenet_first"

        ####################################################
        #           Frozen Feature Extractor (ffe)
        print("Frozen Feature Extractor test:")
        args.num_poisons = 250
        args.trainset_size = 50000
        args.val_period = 20
        args.optimizer = "SGD"
        args.lr = 0.01
        args.lr_schedule = [30]
        args.epochs = 40

        args.end2end = False

        # white-box attack
        args.output = os.path.join(out_dir, "ffe-wb")
        args.model = "vgg16"
        args.model_path = whitebox_modelpath
        poison_test.main(args)

        # grey box attack
        args.model = "vgg16"
        args.model_path = greybox_modelpath
        args.output = os.path.join(out_dir, "ffe-gb")
        poison_test.main(args)

        # black box attacks
        args.output = os.path.join(out_dir, "ffe-bb")

        args.model = "resnet34"
        args.model_path = blackbox_modelpath[0]
        poison_test.main(args)

        args.model_path = blackbox_modelpath[1]
        args.model = "mobilenet_v2"
        poison_test.main(args)
        ####################################################

        # ####################################################
        # #           End-To-End Fine Tuning (e2e)
        # print("End-To-End Fine Tuning test:")
        # args.num_poisons = 25
        # args.trainset_size = 2500
        # args.val_period = 20
        # args.optimizer = "SGD"
        # args.lr = 0.01
        # args.lr_schedule = [30]
        # args.epochs = 40
        #
        # args.end2end = True
        #
        # # white-box attack
        # args.output = os.path.join(out_dir, "e2e-wb")
        # args.model = "resnet18"
        # args.model_path = whitebox_modelpath
        # poison_test.main(args)
        #
        # # grey box attack
        # args.model = "resnet18"
        # args.model_path = greybox_modelpath
        # args.output = os.path.join(out_dir, "e2e-gb")
        # poison_test.main(args)
        #
        # # black box attacks
        # args.output = os.path.join(out_dir, "e2e-bb")
        #
        # args.model = "MobileNetV2"
        # args.model_path = blackbox_modelpath[0]
        # poison_test.main(args)
        #
        # args.model = "VGG11"
        # args.model_path = blackbox_modelpath[1]
        # poison_test.main(args)
        # ####################################################

    else:
        print(
            f"Testing poisons from {args.poisons_path}, in the from scratch training setting...\n".format()
        )

        ####################################################
        #           From Scratch Training (fst)
        args.dataset = "tinyimagenet_all"
        args.num_poisons = 2500
        args.trainset_size = 100000
        args.val_period = 20
        args.optimizer = "SGD"
        args.lr = 0.1
        args.lr_schedule = [100, 150]
        args.epochs = 200
        args.model_path = None
        args.output = os.path.join(out_dir, "fst")

        args.model = "vgg16"
        poison_test.main(args)

        # args.model = "mobilenet_v2"
        # poison_test.main(args)
        #
        # args.model = "resnet34"
        # poison_test.main(args)
        ####################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")
    parser.add_argument(
        "--from_scratch", action="store_true", help="Train from scratch with poisons?"
    )
    parser.add_argument(
        "--poisons_path", default="poisons", type=str, help="where are the poisons?"
    )

    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument(
        "--lr_schedule",
        nargs="+",
        default=[15, 30, 40],
        type=int,
        help="how often to decrease lr",
    )
    parser.add_argument(
        "--lr_factor", default=0.1, type=float, help="factor by which to decrease lr"
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of epochs for training"
    )
    parser.add_argument(
        "--model", default="resnet18", type=str, help="model for training"
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
        "--val_period", default=50, type=int, help="print every __ epoch"
    )
    parser.add_argument(
        "--output", default="output_default", type=str, help="output subdirectory"
    )
    parser.add_argument(
        "--model_path", default="", type=str, help="where is the model saved?"
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
    parser.set_defaults(train_augment=True)
    parser.add_argument(
        "--weight_decay", default=0.0002, type=float, help="weight decay coefficient"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="training batch size"
    )
    parser.add_argument(
        "--target_class", default=None, type=int, help="Which class is the target?"
    )
    parser.add_argument(
        "--target_img_idx",
        default=None,
        type=int,
        help="Index of the target image in the clean set.",
    )
    parser.add_argument("--trainset_size", default=None, type=int, help="Trainset size")
    parser.add_argument("--num_poisons", default=None, type=int, help="number of poisons")

    args = parser.parse_args()
    main(args)
