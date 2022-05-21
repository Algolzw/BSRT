import torch
import numpy as np
from tqdm import tqdm

from datasets.burstsr_dataset import BurstSRDataset, flatten_raw_image
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB

def main():
    train_zurich_raw2rgb = ZurichRAW2RGB(root='/data/dataset/ntire21/burstsr/synthetic', split='train')
    train_data = SyntheticBurst(train_zurich_raw2rgb, burst_size=14, crop_sz=384)
    means = []
    stds = []

    for data in tqdm(train_data):
        print(data.shape)
        break


if __name__ == '__main__':
    # if not args.cpu: torch.cuda.set_device(0)
    main()
