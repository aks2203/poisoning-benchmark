import argparse

import pandas as pd
import numpy as np


def get_error(p, n):
    p = np.array(p).reshape(-1, 1)
    err = np.sqrt(p * (1 - p) / n)
    err[p <= 5.0 / n] = np.sqrt(0.5 * (1 - 0.5) / n)
    err[(1 - p) <= 5.0 / n] = np.sqrt(0.5 * (1 - 0.5) / n)
    return err.flatten()


if __name__ == "__main__":
    print("\n Generating results... \n")
    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--attack_name", type=str, default="fc", help="Which attack?")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Which dataset?")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    args = parser.parse_args()

    models = {
        "cifar10": ["ResNet18", "VGG11", "MobileNetV2"],
        "tinyimagenet": ["VGG16", "ResNet34", "MobileNetV2"],
    }[args.dataset.lower()]
    print(args.filepath)
    df = pd.read_csv(args.filepath, header=0)
    df["model"] = df.apply(lambda x: x["model"].split("/")[-1], axis=1)
    df["batch"] = df.apply(lambda x: x["poisons path"].split("/")[-1], axis=1)
    df["attack"] = df.apply(lambda x: x["poisons path"].split("/")[-2], axis=1)
    df = df[["attack", "model", "batch", "poison_acc"]]
    df.drop_duplicates(subset=["batch", "model", "attack"], inplace=True)
    df = df[df["attack"].str.contains(args.attack_name)]

    print(f"  Attack: {args.attack_name}, dataset: {args.dataset}")
    for model in models:
        print(model)
        df_model = df[df["model"].str.contains(model)]
        if not df_model.empty:
            acc = df_model["poison_acc"].mean()
            err = get_error(acc, df_model.shape[0])[0]
            print(
                f"\tModel: {model}, Poison success: {100*acc: .2f} +/- {100*err: .2f} ({df_model.shape[0]} trials)"
            )

            mylist = df_model["batch"]
            if len(mylist) > 1:
                trial_indices = [a for a in range(args.trials)]
                for idx in [int(m) for m in mylist]:
                    trial_indices.remove(idx)
                print(f"\tModel: {model}, \n\tTrials not yet complete: {trial_indices}\n")
