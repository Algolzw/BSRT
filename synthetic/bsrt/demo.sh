#!/usr/bin/env bash


python main.py --n_GPUs 8 --print_every 40 --lr 0.00003 --decay 50-100 --save bsrt_tiny --model BSRT --fp16 --model_level S --swinfeature --batch_size 32 --burst_size 14 --patch_size 256
python main.py --n_GPUs 8 --print_every 40 --lr 0.00003 --decay 50-100 --save bsrt_large --model BSRT --fp16 --model_level L --swinfeature --batch_size 16 --burst_size 14 --patch_size 256

# python test_synburst.py --n_GPUs 1 --model BSRT --model_level S --fp16 --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_tiny/bsrt_best_epoch.pth --root /data/dataset/ntire21/burstsr/synthetic
# python test_synburst.py --n_GPUs 1 --model BSRT --model_level L --fp16 --swinfeature --burst_size 14 --patch_size 384 --pre_train ../train_log/bsrt/real_models/bsrt_large/bsrt_synburst.pth --root /data/dataset/ntire21/burstsr/synthetic
