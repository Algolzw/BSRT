import torch.nn.functional as F
import cv2
from datasets.synthetic_burst_val_set import SyntheticBurstVal
import torch
import numpy as np
import os


class SimpleBaseline:
    def __init__(self):
        pass

    def __call__(self, burst):
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')
        return burst_rgb


def main():
    dataset = SyntheticBurstVal('PATH_TO_SyntheticBurstVal')
    out_dir = 'PATH_WHERE_RESULTS_ARE_SAVED'

    # TODO Set your network here
    net = SimpleBaseline()

    device = 'cuda'
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(len(dataset)):
        burst, burst_name = dataset[idx]

        burst = burst.to(device).unsqueeze(0)

        with torch.no_grad():
            net_pred = net(burst)

        # Normalize to 0  2^14 range and convert to numpy array
        net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(np.uint16)

        # Save predictions as png
        cv2.imwrite('{}/{}.png'.format(out_dir, burst_name), net_pred_np)


if __name__ == '__main__':
    main()
