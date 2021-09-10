# Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks

**Updated to include new benchmarks on TinyImageNet dataset (November 2020)**

This repository is the official implementation of [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks](https://arxiv.org/abs/2006.12557). 

### CIFAR-10
##### Transfer Learning

| Attack                        | White-box (%)      | Black-box (%)|
| ------------------            |-------------------:|-------------:|
|Feature Collision              | 22.0               | 7.0          |
|Convex Polytope                | 33.0               | 7.0          |
|Bullseye Polytope              | 85.0               | 8.5          |
|Clean Label Backdoor           | 5.0                | 6.5          |
|Hidden Trigger Backdoor        | 10.0               | 9.5          |

    
##### From Scratch Training

| Attack                    | ResNet-18 (%)     | MobileNetV2 (%)   | VGG11 (%) | Average (%)|
| --------------------------| -----------------:|------------------:|----------:|-----------:|
|Feature Collision          |  0                |  1                |  3        |  1.33      |   
|Convex Polytope            |  0                |  1                |  1        |  0.67      |   
|Bullseye Polytope          |  3                |  3                |  1        |  2.33      |   
|Witches' Brew              |  45               |  25               |  8        |  26.00     |   
|Clean Label Backdoor       |  0                |  1                |  2        |  1.00      | 
|Hidden Trigger Backdoor    |  0                |  4                |  1        |  2.67      | 

***

### TinyImageNet
##### Transfer Learning

| Attack                        | White-box (%)      | Black-box (%)|
| ------------------            |-------------------:|-------------:|
|Feature Collision              | 49.0               | 32.0         |
|Convex Polytope                | 14.0               | 1.0          |
|Bullseye Polytope              | 100.0              | 10.5         |
|Clean Label Backdoor           | 3.0                | 1.0          |
|Hidden Trigger Backdoor        | 3.0                | 0.5          |
    
##### From Scratch Training

| Attack                    | VGG11 (%) |
| --------------------------|----------:|
|Feature Collision          |  4        |  
|Convex Polytope            |  0        |  
|Bullseye Polytope          |  44       |  
|Witches' Brew              |  32       |  
|Clean Label Backdoor       |  0        |
|Hidden Trigger Backdoor    |  0        |

###### For more information on each attack consult [our paper](https://arxiv.org/abs/2006.12557) and the original sources listed there.

---

# Getting Started:
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Then download the [TinyImageNet Dataset](https://tiny-imagenet.herokuapp.com/). (Additionally available on our [drive](https://drive.google.com/drive/folders/1MMebJznKStXcFT31MKyyec2GMWcsrwtP?usp=sharing)). In [learning_module.py](learning_module.py), change the line
```
TINYIMAGENET_ROOT = "/fs/cml-datasets/tiny_imagenet"
```
accordingly, to point to the unzipped TinyImageNet directory. (It is left in this repo to match our filesystem, and will likely not work with yours.)

## Pre-trained Models

Pre-trained checkpoints used in this benchmark can be downloaded from [here](https://drive.google.com/drive/folders/1MMebJznKStXcFT31MKyyec2GMWcsrwtP?usp=sharing). They should be copied into the [pretrained_models](pretrained_models) folder (which is empty until downloaded models are added).

---
## Testing

To test a model, run:

```test
python test_model.py --model <model> --model_path <path_to_model_file> 
```    
See the code for additional optional arguments.

## Crafting Poisons With Our Setups
See [How To](how_to.md) for full details and sample code.

## Evaluating A Single Batch of Poison Examples
We have left one sample folder of poisons in poison_examples.
```eval
python poison_test.py --model <model> --model_path <model_path> --poisons_path <path_to_poisons_dir>
```
This allows users to test their poisons in a variety of settings, not only the benchmark setups. See the file [poison_test.py](poison_test.py) for a comprehensive list of arguments.

## Benchmarking A Backdoor or Triggerless Attack
To compute benchmark scores, craft 100 batches of poisons using the setup pickles (for transfer learning: poison_setups_transfer_learning.pickle, for from-scratch training: poison_setups_from_scratch.pickle), and run the following. 

*Important Note:* In order to be on the leaderboard, new submissions must host their poisoned datasets online for public access, so results can be corroborated without producing new poisons. Consider a Dropbox or GoogleDrive folder with all 100 batches of poisons.

For one trial of transfer learning poisons:
```eval
python benchmark_test.py --poisons_path <path_to_poison_directory>  --dataset <dataset>
```

For one trial of from-scratch training poisons:
```eval
python benchmark_test.py --poisons_path <path_to_poison_directory> --dataset <dataset> --from_scratch
```

To benchmark 100 batches of poisons, run
```eval
bash benchmark_all.sh <path_to_directory_with_100_batches> 
``` 
or
```eval
bash benchmark_all.sh <path_to_directory_with_100_batches> from_scratch
``` 
