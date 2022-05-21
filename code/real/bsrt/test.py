import torch.nn.functional as F
import cv2

import torch
import numpy as np
import os
from tqdm import tqdm


from datasets.realworld_burst_test_set import RealWorldBurstTest
from datasets.burstsr_dataset import flatten_raw_image_batch, pack_raw_image_batch
import model

import utility
from option import args

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import time


checkpoint = utility.checkpoint(args)

def main_worker(local_rank, nprocs, args):
    device = 'cuda'
    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)

    dataset = RealWorldBurstTest(args.root)
    out_dir = 'bsrt_realworld'
    os.makedirs(out_dir, exist_ok=True)

    _model = model.Model(args, checkpoint)

    tt = []
    for idx in tqdm(range(len(dataset))):
        burst, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        burst = burst.to(device).unsqueeze(0)

        with torch.no_grad():
            tic = time.time()
            sr = _model(burst, 0).float()
            toc = time.time()
            tt.append(toc-tic)

        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (sr.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
        cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), net_pred_np)

    print('avg time: {:.4f}'.format(np.mean(tt)))
    utility.cleanup()

def main():
    mp.spawn(main_worker, nprocs=1, args=(1, args))

if __name__ == '__main__':
    main()


















