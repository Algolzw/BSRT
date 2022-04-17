import os
import torch
import cv2
import numpy as np
import pickle as pkl


class SyntheticBurstVal(torch.utils.data.Dataset):
    """ Synthetic burst validation set introduced in [1]. The validation burst have been generated using a
    synthetic data generation pipeline. The dataset can be downloaded from
    https://data.vision.ee.ethz.ch/bhatg/SyntheticBurstVal.zip

    [1] Deep Burst Super-Resolution. Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. CVPR 2021
    """
    def __init__(self, root=None, initialize=True):
        """
        args:
            root - Path to root dataset directory
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        self.root = os.path.join(root, 'val')
        self.burst_list = list(range(300))
        self.burst_size = 14

    def initialize(self):
        pass

    def __len__(self):
        return len(self.burst_list)

    def _read_burst_image(self, index, image_id):
        im = cv2.imread('{}/bursts/{:04d}/im_raw_{:02d}.png'.format(self.root, index, image_id), cv2.IMREAD_UNCHANGED)
        im_t = torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1).float() / (2**14)

        return im_t

    def _read_gt_image(self, index):
        gt = cv2.imread('{}/gt/{:04d}/im_rgb.png'.format(self.root, index), cv2.IMREAD_UNCHANGED)
        gt_t = (torch.from_numpy(gt.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float()
        return gt_t

    def _read_meta_info(self, index):
        with open('{}/gt/{:04d}/meta_info.pkl'.format(self.root, index), "rb") as input_file:
            meta_info = pkl.load(input_file)

        return meta_info

    def __getitem__(self, index):
        """ Generates a synthetic burst
        args:
            index: Index of the burst

        returns:
            burst: LR RAW burst, a torch tensor of shape
                   [14, 4, 48, 48]
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            gt : Ground truth linear image
            meta_info: Meta info about the burst which can be used to convert gt to sRGB space
        """
        burst_name = '{:04d}'.format(index)
        burst = [self._read_burst_image(index, i) for i in range(self.burst_size)]
        burst = torch.stack(burst, 0)

        gt = self._read_gt_image(index)
        meta_info = self._read_meta_info(index)
        meta_info['burst_name'] = burst_name
        return burst, gt, meta_info
