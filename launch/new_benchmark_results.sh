#bash benchmark_all.sh /nfshomes/avi1/scratch/bppoison_shortcuts/
#bash benchmark_all.sh ../cifar10_poisons/cp_poisons/num_poisons\=25/
#bash benchmark_all.sh ../cifar10_poisons/fc_poisons/num_poisons\=25/
#bash benchmark_all.sh ../cifar10_poisons/htbd_poisons/num_poisons\=25/
#bash benchmark_all.sh ../cifar10_poisons/clbd_poisons/num_poisons\=25/

bash benchmark_tiny.sh /nfshomes/avi1/scratch/poisoning-benchmark/poison_examples/tinyimagenet_transfer/bp_poisons/num_poisons=250/
bash benchmark_tiny.sh /nfshomes/avi1/scratch/poisoning-benchmark/poison_examples/tinyimagenet_transfer/cp_poisons/num_poisons=250/
bash benchmark_tiny.sh /nfshomes/avi1/scratch/poisoning-benchmark/poison_examples/tinyimagenet_transfer/clbd_poisons/num_poisons=250/
bash benchmark_tiny.sh /nfshomes/avi1/scratch/poisoning-benchmark/poison_examples/tinyimagenet_transfer/fc_poisons/num_poisons=250/
bash benchmark_tiny.sh /nfshomes/avi1/scratch/poisoning-benchmark/poison_examples/tinyimagenet_transfer/htbd_poisons/num_poisons=250/

