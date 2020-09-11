# my_dpc


### training from scratch

python train.py --gpu 0 --net resnet18 --dataset phoenix14 --batch_size 32 --img_dim 128 --epochs 300


### resume training with phoenix dataset from the pretrained k400 model

python train.py --gpu 0 --net resnet18 --dataset phoenix14 --batch_size 32 --img_dim 128 --epochs 300 --resume ./log_tmp/k400_128_r18_dpc-rnn.pth.tar
