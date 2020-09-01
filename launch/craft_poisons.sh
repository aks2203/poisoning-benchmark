#python poison_crafting/craft_poisons_fc.py --normalize --dataset tinyimagenet --poison_setups scaling_setups/tinyimagenet/num_poisons=5/setups.pickle
#python poison_test.py --poisons_path poison_examples/fc_poisons --dataset tinyimagenet --model resnet34 --epochs 2

python poison_crafting/craft_poisons_cp.py --normalize --dataset tinyimagenet --poison_setups scaling_setups/tinyimagenet/num_poisons=5/setups.pickle
python poison_test.py --poisons_path poison_examples/cp_poisons --dataset tinyimagenet --model resnet34 --epochs 2

python poison_crafting/craft_poisons_clbd.py --normalize --dataset tinyimagenet --poison_setups scaling_setups/tinyimagenet/num_poisons=5/setups.pickle
python poison_test.py --poisons_path poison_examples/clbd_poisons --dataset tinyimagenet --model resnet34 --epochs 2

python poison_crafting/craft_poisons_htbd.py --normalize --dataset tinyimagenet --poison_setups scaling_setups/tinyimagenet/num_poisons=5/setups.pickle
python poison_test.py --poisons_path poison_examples/htbd_poisons --dataset tinyimagenet --model resnet34 --epochs 2
