# Crafting Poisons

This subdirectory contains our adapted implementation of the Backdoor and Data Poisoning Attacks mentioned in the [paper](https://arxiv.org/abs/2006.12557).

## Craft Poisons Using Our Implementation

All the defaults are set to the values that were used during the generating the poisons. To generate the poisons, run:

```crafting
python craft_poisons_<attack_name>.py
```
```attack_name``` can take five values here; bp, cp, clbd, fc, htbd.
