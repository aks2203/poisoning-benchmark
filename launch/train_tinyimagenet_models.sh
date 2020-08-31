python train_model.py --dataset tinyimagenet --seed 1000 --model vgg16 --lr 0.01 --lr_schedule 100 150 --lr_factor 0.1 --epochs 200 --optimizer SGD --trainset_size 100000 --val_period 25 --output tinyimagenet_output --checkpoint tinyimagenet_checkpoints --save_net --normalize --train_augment --no-test_augment
python train_model.py --dataset tinyimagenet --seed 1000 --model resnet34 --lr 0.01 --lr_schedule 100 150 --lr_factor 0.1 --epochs 200 --optimizer SGD --trainset_size 100000 --val_period 25 --output tinyimagenet_output --checkpoint tinyimagenet_checkpoints --save_net --normalize --train_augment --no-test_augment
python train_model.py --dataset tinyimagenet --seed 1000 --model mobilenet_v2 --lr 0.01 --lr_schedule 100 150 --lr_factor 0.1 --epochs 200 --optimizer SGD --trainset_size 100000 --val_period 25 --output tinyimagenet_output --checkpoint tinyimagenet_checkpoints --save_net --normalize --train_augment --no-test_augment

python train_model.py --dataset tinyimagenet --seed 1001 --model vgg16 --lr 0.01 --lr_schedule 100 150 --lr_factor 0.1 --epochs 200 --optimizer SGD --trainset_size 100000 --val_period 25 --output tinyimagenet_output --checkpoint tinyimagenet_checkpoints --save_net --normalize --train_augment --no-test_augment
python train_model.py --dataset tinyimagenet --seed 1001 --model resnet34 --lr 0.01 --lr_schedule 100 150 --lr_factor 0.1 --epochs 200 --optimizer SGD --trainset_size 100000 --val_period 25 --output tinyimagenet_output --checkpoint tinyimagenet_checkpoints --save_net --normalize --train_augment --no-test_augment
python train_model.py --dataset tinyimagenet --seed 1001 --model mobilenet_v2 --lr 0.01 --lr_schedule 100 150 --lr_factor 0.1 --epochs 200 --optimizer SGD --trainset_size 100000 --val_period 25 --output tinyimagenet_output --checkpoint tinyimagenet_checkpoints --save_net --normalize --train_augment --no-test_augment
