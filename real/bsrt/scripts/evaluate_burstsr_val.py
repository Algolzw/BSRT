import torch.nn.functional as F
from datasets.burstsr_dataset import BurstSRDataset
from utils.metrics import AlignedPSNR
from pwcnet.pwcnet import PWCNet

root = '/data/dataset/ntire21/burstsr/real/NTIRE/burstsr_dataset'

class SimpleBaseline:
    def __init__(self):
        pass

    def __call__(self, burst):
        burst_rgb = burst[:, 0, [0, 1, 3]]
        burst_rgb = burst_rgb.view(-1, *burst_rgb.shape[-3:])
        burst_rgb = F.interpolate(burst_rgb, scale_factor=8, mode='bilinear')
        return burst_rgb


def main():
    # Load dataset
    dataset = BurstSRDataset(root=root,
                             split='val', burst_size=14, crop_sz=80, random_flip=False)

    # TODO Set your network here
    net = SimpleBaseline()

    device = 'cuda'

    # Load alignment network, used in AlignedPSNR
    alignment_net = PWCNet(load_pretrained=True,
                           weights_path='PATH_TO_PWCNET_WEIGHTS')
    alignment_net = alignment_net.to(device)
    aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net, boundary_ignore=40)

    scores_all = []
    for idx in range(len(dataset)):
        burst, frame_gt, meta_info_burst, meta_info_gt = dataset[idx]
        burst = burst.unsqueeze(0).to(device)
        frame_gt = frame_gt.unsqueeze(0).to(device)

        net_pred = net(burst)

        # Calculate Aligned PSNR
        score = aligned_psnr_fn(net_pred, frame_gt, burst)

        scores_all.append(score)

    mean_psnr = sum(scores_all) / len(scores_all)

    print('Mean PSNR is {:0.3f}'.format(mean_psnr.item()))


if __name__ == '__main__':
    main()
