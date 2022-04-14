from model import common

import torch.nn as nn

class Discriminator(nn.Module):
    '''
        output is not normalized
    '''
    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()

        in_channels = args.n_colors
        out_channels = 32
        depth = 6

        def _block(_in_channels, _out_channels, stride=1):

            Conv = nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                )

            if gan_type == 'SNGAN':
                return nn.Sequential(
                            spectral_norm(Conv),
                            nn.BatchNorm2d(_out_channels),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            else:
                return nn.Sequential(
                    Conv,
                    nn.BatchNorm2d(_out_channels),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            # if i % 2 == 1:
            #     stride = 1
            #     out_channels *= 2
            # else:
            out_channels *= 2
            stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))

        patch_size = args.patch_size // 2**(depth-1)

        # print(out_channels, patch_size)

        m_classifier = [
            nn.Flatten(),
            nn.Linear(out_channels*patch_size**2, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        # print(features.shape)
        output = self.classifier(features)

        return output

