
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

from torchsummaryX import summary


checkpoint = utility.checkpoint(args)


def main():
    mp.spawn(main_worker, nprocs=1, args=(1, args))


def main_worker(local_rank, nprocs, args):
    cudnn.benchmark = True
    args.local_rank = local_rank
    utility.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)

    dataset = BurstSRDataset(root=args.root, burst_size=14, crop_sz=80, split='val')
    out_dir = 'val/bsrt_real'

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

    os.makedirs(out_dir, exist_ok=True)

    tt = []
    psnrs, ssims, lpipss = [], [], []
    for idx in tqdm(range(len(dataset))):
        burst_, gt, meta_info_burst, meta_info_gt = dataset[idx]
        burst_ = burst_.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()
        # burst = flatten_raw_image_batch(burst_)
        name = meta_info_burst['burst_name']

        with torch.no_grad():
            tic = time.time()
            sr = _model(burst_, 0).float()
            toc = time.time()
            tt.append(toc-tic)

            # sr_int = (sr.clamp(0.0, 1.0) * 2 ** 14).short()
            # sr = sr_int.float() / (2 ** 14)

            psnr, ssim, lpips = aligned_psnr_fn(sr, gt, burst_)
            psnrs.append(psnr.item())
            ssims.append(ssim.item())
            lpipss.append(lpips.item())

        # lrs = burst_[0]
        # os.makedirs(f'{out_dir}/{name}', exist_ok=True)
        # for i, lr in enumerate(lrs):
        #     # print(lr[[0, 1, 3],...].shape)
        #     lr = postprocess_fn.process(lr[[0, 1, 3],...], meta_info_burst)
        #     lr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('{}/{}/{:2d}.png'.format(out_dir, name, i), lr)

        # gt = postprocess_fn.process(gt[0], meta_info_burst)
        # gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('{}/{}_gt.png'.format(out_dir, name), gt)

        # sr_ = postprocess_fn.process(sr[0], meta_info_burst)
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
