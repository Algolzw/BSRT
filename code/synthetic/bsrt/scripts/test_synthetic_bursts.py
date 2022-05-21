import torch.nn.functional as F
import cv2
from datasets.synthetic_burst_train_set import SyntheticBurst
from torch.utils.data.dataloader import DataLoader
from utils.metrics import PSNR
from utils.postprocessing_functions import SimplePostProcess
from utils.data_format_utils import convert_dict
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB


def main():
    zurich_raw2rgb = ZurichRAW2RGB(root='PATH_TO_ZURICH_RAW_TO_RGB', split='test')
    dataset = SyntheticBurst(zurich_raw2rgb, burst_size=3, crop_sz=256)

    data_loader = DataLoader(dataset, batch_size=2)

    # Function to calculate PSNR. Note that the boundary pixels (40 pixels) will be ignored during PSNR computation
    psnr_fn = PSNR(boundary_ignore=40)

    # Postprocessing function to obtain sRGB images
    postprocess_fn = SimplePostProcess(return_np=True)

    for d in data_loader:
        burst, frame_gt, flow_vectors, meta_info = d

        # A simple baseline which upsamples the base image using bilinear upsampling
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')

        # Calculate PSNR
        score = psnr_fn(burst_rgb, frame_gt)

        print('PSNR is {:0.3f}'.format(score))

        meta_info = convert_dict(meta_info, burst.shape[0])

        # Apply simple post-processing to obtain RGB images
        pred_0 = postprocess_fn.process(burst_rgb[0], meta_info[0])
        gt_0 = postprocess_fn.process(frame_gt[0], meta_info[0])

        pred_0 = cv2.cvtColor(pred_0, cv2.COLOR_RGB2BGR)
        gt_0 = cv2.cvtColor(gt_0, cv2.COLOR_RGB2BGR)

        # Visualize input, ground truth
        cv2.imshow('Input (Demosaicekd + Upsampled)', pred_0)
        cv2.imshow('GT', gt_0)

        input_key = cv2.waitKey(0)
        if input_key == ord('q'):
            return


if __name__ == '__main__':
    main()
