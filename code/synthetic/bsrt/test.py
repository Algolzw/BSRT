
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import utility
from option import args

from datasets.synthetic_burst_test_set import SyntheticBurstTest
from datasets.burstsr_dataset import flatten_raw_image_batch, pack_raw_image_batch
import model

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import time


checkpoint = utility.checkpoint(args)

def ttaup(burst):
    burst0 = flatten_raw_image_batch(burst) # B, T, C, H, W
    burst1 = utility.bayer_aug(burst0, flip_h=False, flip_w=False, transpose=True)
    burst0 = pack_raw_image_batch(burst0)
    burst1 = pack_raw_image_batch(burst1)

    return [burst0, burst1]


def ttadown(bursts):
    burst0 = bursts[0]
    burst1 = bursts[1].permute(0, 1, 3, 2)
    out = (burst0 + burst1) / 2
    return out


def main():
    mp.spawn(main_worker, nprocs=1, args=(1, args))


def main_worker(local_rank, nprocs, args):
    device = 'cuda'
    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)

    dataset = SyntheticBurstTest(args.root)
    out_dir = 'bsrt_synburst'
    os.makedirs(out_dir, exist_ok=True)

    _model = model.Model(args, checkpoint)

    tt = []
    for idx in tqdm(range(len(dataset))):
        burst, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        burst = burst.to(device).unsqueeze(0)
        bursts = ttaup(burst)

        srs = []
        with torch.no_grad():
            for x in bursts:
                tic = time.time()
                sr = _model(x, 0)
                toc = time.time()
                tt.append(toc-tic)
                srs.append(sr)

        sr = ttadown(srs)
        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (sr.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)
        cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), net_pred_np)

    print('avg time: {:.4f}'.format(np.mean(tt)))
    utility.cleanup()


if __name__ == '__main__':
    main()
