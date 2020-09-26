import argparse

import pandas as pd
import numpy as np


def get_error(p, n):
    p = np.array(p).reshape(-1, 1)
    err = np.sqrt(p * (1 - p) / n)
    err[p <= 5.0/n] = np.sqrt(0.5 * (1 - 0.5) / n)
    err[(1-p) <= 5.0/n] = np.sqrt(0.5 * (1 - 0.5) / n)
    return err.flatten()


def scaling_results(results_path, attackname):
    df = pd.read_csv(results_path, header=0)
    df.insert(1, "Attack", attackname, True)
    df.insert(1, "Train Set Size", 100*df['num poisons'])
    print([df.loc[df['num poisons'] == n].shape for n in [5, 10, 25, 50, 100, 200, 500]])
    numbers_of_poisons = sorted(list(set(df['num poisons'])))
    nums = []
    accs = []
    for num in numbers_of_poisons:
        print(num, ' poison examples: ')
        small_df = df.loc[df['num poisons'] == num]
        nums.append(num)
        accs.append((sum(small_df['poison_acc'])/small_df.shape[0]))
        # accs.append((sum(small_df['flip_acc'])/small_df.shape[0]))
        error = get_error(accs[-1], small_df.shape[0])
        print('success rate on  clean target: %.2f +/- %.2f' % (100*accs[-1], 100*error))
        print(' ')
    accs = np.array(accs)
    print("______________________")
    return df, accs


if __name__ == "__main__":
    print("\n Generating results... \n")
    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--attack_name", type=str, default="fc", help="Which attack?")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Which dataset?")
    args = parser.parse_args()

    models = {"cifar10": ["resnet18", "vgg11", "mobilenetv2"],
              "tinyimagenet": ["vgg16", "resnet34", "mobilenet_v2"]}[args.dataset]
    print(args.filepath)
    df = pd.read_csv(args.filepath, header=0)
    df = df[df['poisons path'].str.contains(args.attack_name)].tail(400)

    # print(f"File path: {str(args.filepath)}")
    print(f"  Attack: {args.attack_name}, dataset: {args.dataset}")
    for model in models:
        df_model = df[df["model"].str.contains(model)]
        if not df_model.empty:
            acc = df_model['poison_acc'].mean()
            err = get_error(acc, df_model.shape[0])[0]
            print(f"\tModel: {model}, Poison success: {100*acc: .2f} +/- {100*err: .2f} ({df_model.shape[0]} trials)")

    # print("\nDone.\n")
