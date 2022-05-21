
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import utility
from option import args

from utils.postprocessing_functions import SimplePostProcess
from datasets.burstsr_dataset import flatten_raw_image_batch, pack_raw_image, pack_raw_image_batch
from datasets.synthetic_burst_val_set import SyntheticBurstVal
from utils.metrics import PSNR
from utils.data_format_utils import convert_dict
from data_processing.camera_pipeline import demosaic
import model

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import time

# from torchsummaryX import summary


checkpoint = utility.checkpoint(args)

def ttaup(burst):
    # burst0 = flatten_raw_image_batch(burst) # B, T, C, H, W
    # burst1 = utility.bayer_aug(burst0, flip_h=False, flip_w=False, transpose=True)
    # burst1 = pack_raw_image_batch(burst1)
    return [burst]

def ttadown(bursts):
    burst0 = bursts[0]
    # burst1 = bursts[1].permute(0, 1, 3, 2)
    # out = (burst0 + burst1) / 2
    out = burst0
    return out

def main():
    mp.spawn(main_worker, nprocs=1, args=(1, args))


def main_worker(local_rank, nprocs, args):
    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)

    dataset = SyntheticBurstVal(root=args.root)
    out_dir = 'val/bsrt_synburst'

    _model = model.Model(args, checkpoint)

    for param in _model.parameters():
        param.requires_grad = False

    psnr_fn = PSNR(boundary_ignore=40)

    postprocess_fn = SimplePostProcess(return_np=True)

    os.makedirs(out_dir, exist_ok=True)

    tt = []
    psnrs, ssims, lpipss = [], [], []
    for idx in tqdm(range(len(dataset))):
        burst_, gt, meta_info = dataset[idx]
        burst_ = burst_.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()
        name = meta_info['burst_name']

        bursts = ttaup(burst_)

        srs = []
        with torch.no_grad():
            for x in bursts:
                tic = time.time()
                sr = _model(x, 0).float()
                toc = time.time()
                tt.append(toc-tic)
                srs.append(sr)

            sr = ttadown(srs)

            # sr_int = (sr.clamp(0.0, 1.0) * 2 ** 14).short()
            # sr = sr_int.float() / (2 ** 14)

        psnr, ssim, lpips = psnr_fn(sr, gt)
        psnrs.append(psnr.item())
        ssims.append(ssim.item())
        lpipss.append(lpips.item())


        # lrs = burst_[0]
        # os.makedirs(f'{out_dir}/{name}', exist_ok=True)
        # for i, lr in enumerate(lrs):
        #     # print(lr[[0, 1, 3],...].shape)
        #     lr = postprocess_fn.process(lr[[0, 1, 3],...], meta_info)
        #     lr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('{}/{}/{:2d}.png'.format(out_dir, name, i), lr)

        # gt = postprocess_fn.process(gt[0], meta_info)
        # gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('{}/{}_gt.png'.format(out_dir, name), gt)

        # sr_ = postprocess_fn.process(sr[0], meta_info)
        # sr_ = cv2.cvtColor(sr_, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('{}/{}_bsrt.png'.format(out_dir, name), sr_)

        del burst_
        del sr
        del gt


    print(f'avg PSNR: {np.mean(psnrs):.6f}')
    print(f'avg SSIM: {np.mean(ssims):.6f}')
    print(f'avg LPIPS: {np.mean(lpipss):.6f}')
    print(f' avg time: {np.mean(tt):.6f}')

    # utility.cleanup()


if __name__ == '__main__':
    main()
