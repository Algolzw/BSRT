
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import utility
from option import args
import torchvision.utils as tvutils
from pwcnet.pwcnet import PWCNet

from utils.postprocessing_functions import BurstSRPostProcess
from datasets.burstsr_dataset import BurstSRDataset, flatten_raw_image_batch, pack_raw_image
from utils.metrics import AlignedPSNR
from utils.data_format_utils import convert_dict
from data_processing.camera_pipeline import demosaic
import model

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import time


checkpoint = utility.checkpoint(args)


def main():
    mp.spawn(main_worker, nprocs=1, args=(1, args))


def main_worker(local_rank, nprocs, args):
    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)

    dataset = BurstSRDataset(root=args.root, burst_size=14, crop_sz=80, split='val')
    # out_dir = 'val/ebsr_real'

    _model = model.Model(args, checkpoint)

    for param in _model.parameters():
        param.requires_grad = False

    alignment_net = PWCNet(load_pretrained=True,
                           weights_path='./pwcnet/pwcnet-network-default.pth')
    alignment_net = alignment_net.to('cuda')
    for param in alignment_net.parameters():
        param.requires_grad = False

    aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)

    postprocess_fn = BurstSRPostProcess(return_np=True)

    # os.makedirs(out_dir, exist_ok=True)

    tt = []
    psnrs, ssims, lpipss = [], [], []
    for idx in tqdm(range(len(dataset))):
        burst, gt, meta_info_burst, meta_info_gt = dataset[idx]
        burst = burst.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()

        with torch.no_grad():
            tic = time.time()
            sr = _model(burst, 0).float()
            toc = time.time()
            tt.append(toc-tic)

            # sr_int = (sr.clamp(0.0, 1.0) * 2 ** 14).short()
            # sr = sr_int.float() / (2 ** 14)

            psnr, ssim, lpips = aligned_psnr_fn(sr, gt, burst)
            psnrs.append(psnr.item())
            ssims.append(ssim.item())
            lpipss.append(lpips.item())

        # os.makedirs(f'{out_dir}/{idx}', exist_ok=True)
        # sr_ = postprocess_fn.process(sr[0], meta_info_burst)
        # sr_ = cv2.cvtColor(sr_, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('{}/{}_sr.png'.format(out_dir, idx), sr_)

        del burst
        del sr
        del gt


    print(f'avg PSNR: {np.mean(psnrs):.6f}')
    print(f'avg SSIM: {np.mean(ssims):.6f}')
    print(f'avg LPIPS: {np.mean(lpipss):.6f}')
    print(f' avg time: {np.mean(tt):.6f}')

    # utility.cleanup()


if __name__ == '__main__':
    main()
