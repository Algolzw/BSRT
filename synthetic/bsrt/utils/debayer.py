import torch
import torch.nn
import torch.nn.functional

class Debayer3x3(torch.nn.Module):
    '''Demosaicing of Bayer images using 3x3 convolutions.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.

    Compared to Debayer2x2 this method does not use upsampling.
    Instead, we identify five 3x3 interpolation kernels that
    are sufficient to reconstruct every color channel at every
    pixel location.

    We convolve the image with these 5 kernels using stride=1
    and a one pixel replication padding. Finally, we gather
    the correct channel values for each pixel location. Todo so,
    we recognize that the Bayer pattern repeats horizontally and
    vertically every 2 pixels. Therefore, we define the correct
    index lookups for a 2x2 grid cell and then repeat to image
    dimensions.

    Note, in every 2x2 grid cell we have red, blue and two greens
    (G1,G2). The lookups for the two greens differ.
    '''

    def __init__(self):
        super(Debayer3x3, self).__init__()

        self.kernels = torch.nn.Parameter(
            torch.tensor([
                [0,0,0],
                [0,1,0],
                [0,0,0],

                [0, 0.25, 0],
                [0.25, 0, 0.25],
                [0, 0.25, 0],

                [0.25, 0, 0.25],
                [0, 0, 0],
                [0.25, 0, 0.25],

                [0, 0, 0],
                [0.5, 0, 0.5],
                [0, 0, 0],

                [0, 0.5, 0],
                [0, 0, 0],
                [0, 0.5, 0],
            ]).view(5,1,3,3), requires_grad=False
        )


        self.index = torch.nn.Parameter(
            torch.tensor([
                # dest channel r
                [0, 3], # pixel is R,G1
                [4, 2], # pixel is G2,B
                # dest channel g
                [1, 0], # pixel is R,G1
                [0, 1], # pixel is G2,B
                # dest channel b
                [2, 4], # pixel is R,G1
                [3, 0], # pixel is G2,B
            ]).view(1,3,2,2), requires_grad=False
        )

    def forward(self, x):
        '''Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        '''
        B,C,H,W = x.shape

        x = torch.nn.functional.pad(x, (1,1,1,1), mode='replicate')
        c = torch.nn.functional.conv2d(x, self.kernels, stride=1)
        rgb = torch.gather(c, 1, self.index.repeat(B,1,H//2,W//2))
        return rgb

class Debayer2x2(torch.nn.Module):
    '''Demosaicing of Bayer images using 2x2 convolutions.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.
    '''

    def __init__(self):
        super(Debayer2x2, self).__init__()

        self.kernels = torch.nn.Parameter(
            torch.tensor([
                [1, 0],
                [0, 0],

                [0, 0.5],
                [0.5, 0],

                [0, 0],
                [0, 1],
            ]).view(3,1,2,2), requires_grad=False
        )

    def forward(self, x):
        '''Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        '''

        x = torch.nn.functional.conv2d(x, self.kernels, stride=2)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

class DebayerSplit(torch.nn.Module):
    '''Demosaicing of Bayer images using 3x3 green convolution and red,blue upsampling.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.
    '''
    def __init__(self):
        super().__init__()

        self.pad = torch.nn.ReflectionPad2d(1)
        self.kernel = torch.nn.Parameter(
            torch.tensor([
                [0,1,0],
                [1,0,1],
                [0,1,0]
            ])[None, None] * 0.25)

    def forward(self, x):
        '''Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        '''
        B,_,H,W = x.shape
        red = x[:, :, ::2, ::2]
        blue = x[:, :, 1::2, 1::2]

        green = torch.nn.functional.conv2d(self.pad(x), self.kernel)
        green[:, :, ::2, 1::2] = x[:, :, ::2, 1::2]
        green[:, :, 1::2, ::2] = x[:, :, 1::2, ::2]

        return torch.cat((
            torch.nn.functional.interpolate(red, size=(H, W), mode='bilinear', align_corners=False),
            green,
            torch.nn.functional.interpolate(blue, size=(H, W), mode='bilinear', align_corners=False)),
            dim=1)