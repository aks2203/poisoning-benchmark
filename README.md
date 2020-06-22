# Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks

This repository is the official implementation of [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks](). 

### Benchmark Scores

#### Frozen Feature Extractor
| Attack                        | White-box (%)   | Grey-box (%)   | Black-box (%)|
| ------------------            |---------------- | -------------- |--------------|
|Featrue Collision              | 16.0            | 7.0            | 3.50         |
|Featrue Collision  Ensembled   | 13.0            | 9.0            | 6.0          |
|Convex Polytope                | 24.0            | 7.0            | 4.5          |
|Convex Polytope Ensembled      | 20.0            | 8.0            | 12.5         |
|Clean Label Backdoor           | 3.0             | 6.0            | 3.5          |
|Hidden Trigger Backdoor        | 2.0             | 4.0            | 4.0          |
    
#### End-to-end Fine-tuning
| Attack                        | White-box (%)     | Grey-box (%)   | Black-box (%) |
| ------------------            |----------------   | -------------- |-----------   |
|Featrue Collision              | 4.0               | 3.0            | 3.5          |
|Featrue Collision  Ensembled   | 7.0               | 4.0            | 5.0          |
|Convex Polytope                | 17.0              | 7.0            | 4.5          |
|Convex Polytope Ensembled      | 14.0              | 4.0            | 10.5         |
|Clean Label Backdoor           | 3.0               | 2.0            | 1.5          |
|Hidden Trigger Backdoor        | 3.0               | 2.0            | 4.0          |

#### From Scratch Training
| Attack                    | ResNet-18 (%)     | MobileNetV2 (%)   | VGG11 (%) | Average (%)|
| --------------------------| --------------    |-----------        |-----------|----------- |
|Featrue Collision          |  0                |  1                |  3        |  1.33      |   
|Convex Polytope            |  0                |  1                |  1        |  0.67      |   
|Clean Label Backdoor       |  0                |  1                |  2        |  1.00      | 
|Hidden Trigger Backdoor    |  0                |  4                |  1        |  2.67      | 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Pre-trained Models

Pre-trained checkpoints used in this benchmark in the [pretrained_models](pretrained_models) folder.


## Testing

To test a model, run:

```test
python test_model.py --model <model> --model_path <path_to_model_file> 
```

## Crafting Poisons With Our Setups
See [How To](how_to.md) for full details and sample code.

## Evaluating A Single Batch of Poison Examples
We have left one sample folder of poisons in poison_examples.
```eval
python poison_test.py --model <model> --model_path <model_path> --poisons_path <path_to_poisons_dir>
```
This allows users to test their poisons in a variety of settings, not only the benchmark setups.

## Benchmarking A Backdoor or Triggerless Attack
To compute benchmark scores, craft 100 batches of poisons using the setup pickles (for transfer learning: poison_setups_transfer_learning.pickle, for from-scratch training: poison_setups_from_scratch.pickle), and run the following. 

For one trial of transfer learning poisons:
```eval
python benchmark_test.py --poisons_path <path_to_poison_directory>
```

For one trial of from-scratch training poisons:
```eval
python benchmark_test.py --poisons_path <path_to_poison_directory> --from_scratch
```

To benchmark 100 batches of poisons, run
```eval
bash benchmark_all.sh <path_to_directory_with_100_batches> 
``` 
or
```eval
bash benchmark_all.sh <path_to_directory_with_100_batches> from_scratch
``` 
