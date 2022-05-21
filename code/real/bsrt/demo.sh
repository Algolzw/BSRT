#!/usr/bin/env bash


python main.py --n_GPUs 8 --print_every 20 --lr 0.00004 --decay 40-80 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 8 --burst_size 14 --patch_size 80 --pre_train ../../synthetic/train_log/bsrt/real_models/bsrt_tiny/bsrt_best_epoch.pth 
# python main.py --n_GPUs 8 --print_every 20 --lr 0.00004 --decay 40-80 --save bsrt_large --model BSRT --fp16 --model_level L --swinfeature --batch_size 8 --burst_size 14 --patch_size 48 --pre_train ../../synthetic/train_log/bsrt/real_models/bsrt_large/bsrt_best_epoch.pth 


# python test_real.py --n_GPUs 1 --model BSRT --model_level S --swinfeature --batch_size 1 --burst_size 14 --patch_size 80 --pre_train ../train_log/bsrt/real_models/bsrt_tiny/bsrtbest_epoch.pth --root /data/dataset/ntire21/burstsr/real
# python test_real.py --n_GPUs 1 --model BSRT --model_level L --swinfeature --batch_size 1 --burst_size 14 --patch_size 80 --pre_train ../train_log/bsrt/real_models/bsrt_large/bsrt_realworld.pth --root /data/dataset/ntire21/burstsr/real