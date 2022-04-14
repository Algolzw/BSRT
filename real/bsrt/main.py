import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import os

import utility
import model
import loss
from option import args
from trainer import Trainer
from datasets.burstsr_dataset import BurstSRDataset, flatten_raw_image
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

checkpoint = utility.checkpoint(args)

def main():
    mp.spawn(main_worker, nprocs=args.n_GPUs, args=(args.n_GPUs, args))


def main_worker(local_rank, nprocs, args):
    # print(local_rank)
    if checkpoint.ok:
        args.local_rank = local_rank
        init_seeds(local_rank+1)
        cudnn.benchmark = True
        utility.setup(local_rank, nprocs)
        torch.cuda.set_device(local_rank)

        batch_size = int(args.batch_size / nprocs)
        train_data = BurstSRDataset(root=args.root,
                                    burst_size=args.burst_size,
                                    crop_sz=args.patch_size, random_flip=True,
                                    center_crop=True, split='train')
        valid_data = BurstSRDataset(root=args.root,
                                    burst_size=14,
                                    crop_sz=80, split='val')

        if local_rank <= 0:
            print(f"train data: {len(train_data)}, test data: {len(valid_data)}")

        if nprocs > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, shuffle=False)
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=args.batch_size,
                                        pin_memory=True, drop_last=True, sampler=train_sampler)  # args.cpus
            valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, num_workers=args.batch_size,
                                        pin_memory=True, drop_last=True, sampler=valid_sampler)  # args.cpus
        else:
            train_sampler = None
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=8,
                                    shuffle=True, pin_memory=True, drop_last=True)  # args.cpus
            valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                    pin_memory=True, drop_last=True)  # args.cpus

        _model = model.Model(args, checkpoint)

        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, train_loader, train_sampler, valid_loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()

        del _model
        del _loss
        del train_loader
        del valid_loader

        # checkpoint.done()

if __name__ == '__main__':
    # if not args.cpu: torch.cuda.set_device(0)
    main()
