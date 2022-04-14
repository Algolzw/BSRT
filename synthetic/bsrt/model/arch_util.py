import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model import common
from model.utils.psconv import PSGConv2d as PSConv2d, PyConv2d


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


###########################

def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True)
    

class ESA(nn.Module):
    def __init__(self, n_feats, conv=conv_layer):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

##########################

class SELayer(nn.Module):
    '''
    SE-block
    '''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock_SE(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, reduction=16):
        super(ResidualBlock_SE, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(3 * nf, nf, 1, padding=0, dilation=1, bias=True)
        self.se = SELayer(nf, reduction)
        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        identity = x
        basic_out = F.relu(self.conv1(x), inplace=True)
        basic_out = self.conv2(basic_out)
        se_out = self.se(basic_out)
        out = torch.cat((identity, basic_out, se_out), 1)
        out = self.conv3(out)
        return out


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

## Spatial Attention (CA) Layer
class SALayer(nn.Module):
    def __init__(self, wn=None):
        super(SALayer,self).__init__()
        self.body = nn.Sequential(
            wn(nn.Conv2d(2, 1, 7, 1, 3, bias=False)),
            nn.Sigmoid()
        )
    def forward(self, x):
        avg_f = torch.mean(x, dim=1, keepdim=True)
        max_f = torch.max(x, dim=1, keepdim=True)[0]
        y = torch.cat([avg_f, max_f], dim=1)
        return self.body(y).expand_as(x) * x


## Channel Attention (CA) Layer
class CALayerV2(nn.Module):
    def __init__(self, n_feat, reduction=16, wn=None):
        super(CALayerV2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                wn(nn.Conv2d(n_feat, n_feat//reduction, 1, padding=0, bias=False)),
                nn.ReLU(inplace=True),
                wn(nn.Conv2d(n_feat//reduction, n_feat, 1, padding=0, bias=False)),
                # nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y1 = self.conv_du(y1)
        y2 = self.conv_du(y2)
        return x * torch.sigmoid(y1+y2)

class DALayer(nn.Module):
    def __init__(self, channel, reduction, wn):
        super(DALayer, self).__init__()
        # global average pooling: feature --> point
        self.ca = CALayer(channel, reduction, wn)
        self.sa = SALayer(wn)
        self.conv = wn(nn.Conv2d(channel*2, channel, 1))

    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(x)
        res = self.conv(torch.cat([ca, sa], dim=1))
        return res + x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction, wn):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                wn(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)),
                nn.ReLU(inplace=True),
                wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, wn,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, da=False):

        super(RCAB, self).__init__()

        expand = 6
        linear = 0.75
        modules_body = []
        # for i in range(2):
        modules_body.append(wn(nn.Conv2d(n_feat, n_feat*expand, 1, bias=bias)))
        modules_body.append(act)
        modules_body.append(wn(nn.Conv2d(n_feat*expand, int(n_feat*linear), 1, bias=bias)))
        modules_body.append(conv(int(n_feat*linear), n_feat, kernel_size, bias=bias))
        if da:
            modules_body.append(DALayer(n_feat, reduction, wn))
        else:
            modules_body.append(CALayer(n_feat, reduction, wn))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks, da=False):
        super(ResidualGroup, self).__init__()
        kernel_size = 3
        res_scale = 1
        reduction = 16

        conv = common.default_conv
        wn = lambda x: torch.nn.utils.weight_norm(x)

        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, wn=wn, bias=True,
                bn=False, act=nn.ReLU(True), res_scale=res_scale, da=da) \
            for _ in range(n_resblocks)]
        modules_body.append(wn(conv(n_feat, n_feat, kernel_size)))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


################################################################
################################################################
################################################################

def make_layer_idx(block, n_layers):
    layers = []
    for i in range(n_layers):
        layers.append(block(idx=i))
    return nn.Sequential(*layers)

## Residual Channel Attention Block (RCAB)
class LRSCRCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, wn,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, da=False, idx=0):
        super(LRSCRCAB, self).__init__()

        expand = 6
        linear = 0.75

        modules_body = [wn(nn.Conv2d(n_feat*(idx+1), n_feat, 1, 1, 0, bias=True))] if idx > 0 else []
        # for i in range(2):
        modules_body.append(wn(nn.Conv2d(n_feat, n_feat*expand, 1, bias=bias)))
        modules_body.append(act)
        modules_body.append(wn(nn.Conv2d(n_feat*expand, int(n_feat*linear), 1, bias=bias)))
        modules_body.append(wn(conv(int(n_feat*linear), n_feat, kernel_size, bias=bias)))
        if da:
            modules_body.append(DALayer(n_feat, reduction, wn))
        else:
            modules_body.append(CALayer(n_feat, reduction, wn))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res  = torch.cat([res, x], dim=1)
        return res


## Residual Channel Attention Block (RCAB)
class LRSCPYRCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, wn,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, da=False, idx=0):
        super(LRSCPYRCAB, self).__init__()

        expand = 6
        linear = 0.75

        modules_body = [wn(nn.Conv2d(n_feat*(idx+1), n_feat, 1, 1, 0, bias=True))] if idx > 0 else []
        # for i in range(2):
        modules_body.append(wn(nn.Conv2d(n_feat, n_feat*expand, 1, bias=bias)))
        modules_body.append(act)
        modules_body.append(wn(nn.Conv2d(n_feat*expand, int(n_feat*linear), 1, bias=bias)))
        modules_body.append(
            PyConv2d(in_channels=int(n_feat*linear),
                out_channels=[n_feat//4, n_feat//4, n_feat//2],
                pyconv_kernels=[3, 5, 7],
                pyconv_groups=[1, 4, 8]))
        if da:
            modules_body.append(DALayer(n_feat, reduction, wn))
        else:
            modules_body.append(CALayer(n_feat, reduction, wn))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res  = torch.cat([res, x], dim=1)
        return res

## Long-Range Skip-connect Residual Group (RG)
class LRSCResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks, da=False, idx=0):
        super(LRSCResidualGroup, self).__init__()
        kernel_size = 3
        res_scale = 1
        reduction = 16

        conv = common.default_conv
        wn = lambda x: torch.nn.utils.weight_norm(x)

        modules_head = [wn(conv(n_feat*(idx+1), n_feat, 1, bias=True))] if idx > 0 else []
        modules_body = [
            LRSCRCAB(
                conv, n_feat, kernel_size, reduction, wn=wn, bias=True,
                bn=False, act=nn.ReLU(True), res_scale=res_scale, da=da, idx=i) \
            for i in range(n_resblocks)]
        modules_body.append(wn(conv(n_feat*(n_resblocks+1), n_feat, kernel_size)))
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res  = torch.cat([res, x], dim=1)
        return res


## Long-Range Skip-connect Residual Group (RG)
class LRSCPSResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks, da=False, idx=0):
        super(LRSCPSResidualGroup, self).__init__()
        kernel_size = 3
        res_scale = 1
        reduction = 16

        conv = PSConv2d
        wn = lambda x: torch.nn.utils.weight_norm(x)

        modules_head = [wn(nn.Conv2d(n_feat*(idx+1), n_feat, 1, 1, 0, bias=True))] if idx > 0 else []
        modules_body = [
            LRSCRCAB(
                conv, n_feat, kernel_size, reduction, wn=wn, bias=True,
                bn=False, act=nn.ReLU(True), res_scale=res_scale, da=da, idx=i) \
            for i in range(n_resblocks)]
        modules_tail = [wn(conv(n_feat*(n_resblocks+1), n_feat, kernel_size))]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        res  = torch.cat([res, x], dim=1)
        return res


## Long-Range Skip-connect Residual Group (RG)
class LRSCPyResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks, da=False, idx=0):
        super(LRSCPyResidualGroup, self).__init__()
        kernel_size = 3
        res_scale = 1
        reduction = 16

        conv = PyConv2d
        wn = lambda x: torch.nn.utils.weight_norm(x)

        modules_head = [wn(nn.Conv2d(n_feat*(idx+1), n_feat, 1, 1, 0, bias=True))] if idx > 0 else []
        modules_body = [
            LRSCPYRCAB(
                conv, n_feat, kernel_size, reduction, wn=wn, bias=True,
                bn=False, act=nn.ReLU(True), res_scale=res_scale, da=da, idx=i) \
            for i in range(n_resblocks)]
        modules_tail = [wn(nn.Conv2d(n_feat*(n_resblocks+1), n_feat, 1))]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        res  = torch.cat([res, x], dim=1)
        return res

class LRSCWideActResBlock(nn.Module):
    def __init__(self, nf=64, idx=0):
        super(LRSCWideActResBlock, self).__init__()
        self.res_scale = 1

        expand = 6
        linear = 0.8
        kernel_size = 3
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act=nn.ReLU(True)
        head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, bias=True))] if idx > 0 else []

        body = []
        body.append(
            wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res  = torch.cat([res, x], dim=1)
        return res

class LRSCPyWideActResBlock(nn.Module):
    def __init__(self, nf=64, idx=0):
        super(LRSCPyWideActResBlock, self).__init__()
        self.res_scale = 1

        expand = 6
        linear = 0.75
        kernel_size = 3
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act=nn.ReLU(True)
        head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, bias=True))] if idx > 0 else []

        body = []
        body.append(
            wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
        body.append(
            PyConv2d(in_channels=int(nf*linear),
                out_channels=[nf//4, nf//4, nf//2],
                pyconv_kernels=[3, 5, 7],
                pyconv_groups=[1, 4, 8]))

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res  = torch.cat([res, x], dim=1)
        return res


## Long-Range Skip-connect Residual Group (RG)
class LRSCPyWideActResGroup(nn.Module):
    def __init__(self, nf, n_resblocks, idx=0):
        super(LRSCPyWideActResGroup, self).__init__()
        kernel_size = 3

        conv = PyConv2d
        wn = lambda x: torch.nn.utils.weight_norm(x)

        modules_head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, 1, 0, bias=True))] if idx > 0 else []
        modules_body = [
            LRSCPyWideActResBlock(nf=nf, idx=i) for i in range(n_resblocks)]
        modules_tail = [wn(nn.Conv2d(nf*(n_resblocks+1), nf, 1))]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        res  = torch.cat([res, x], dim=1)
        return res


## Long-Range Skip-connect Residual Group (RG)
class LRSCWideActResGroup(nn.Module):
    def __init__(self, nf, n_resblocks, idx=0):
        super(LRSCWideActResGroup, self).__init__()
        kernel_size = 3

        conv = PyConv2d
        wn = lambda x: torch.nn.utils.weight_norm(x)

        modules_head = [wn(nn.Conv2d(nf*(idx+1), nf, 1, 1, 0, bias=True))] if idx > 0 else []
        modules_body = [
            LRSCWideActResBlock(nf=nf, idx=i) for i in range(n_resblocks)]
        modules_tail = [wn(nn.Conv2d(nf*(n_resblocks+1), nf, 1))]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        res = self.head(x)
        res = self.body(res)
        res = self.tail(res)
        res  = torch.cat([res, x], dim=1)
        return res

################################################################
################################################################
################################################################


## Residual Channel Attention Block (RCAB)
class PYRCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, wn,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, da=False):
        super(PYRCAB, self).__init__()

        expand = 6
        linear = 0.75
        modules_body = []
        # for i in range(2):
        modules_body.append(wn(nn.Conv2d(n_feat, n_feat*expand, 1, bias=bias)))
        modules_body.append(act)
        modules_body.append(wn(nn.Conv2d(n_feat*expand, int(n_feat*linear), 1, bias=bias)))
        # modules_body.append(conv(, n_feat, kernel_size, bias=bias))
        modules_body.append(PyConv2d(in_channels=int(n_feat*linear),
                out_channels=[n_feat//4, n_feat//4, n_feat//2],
                pyconv_kernels=[3, 5, 7],
                pyconv_groups=[1, 4, 8], bias=bias))
        if da:
            modules_body.append(DALayer(n_feat, reduction, wn))
        else:
            modules_body.append(CALayer(n_feat, reduction, wn))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class PyResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks, da=False):
        super(PyResidualGroup, self).__init__()
        kernel_size = 3
        res_scale = 1
        reduction = 16

        conv = PyConv2d
        wn = lambda x: torch.nn.utils.weight_norm(x)

        modules_body = []
        modules_body = [
            PYRCAB(
                conv, n_feat, kernel_size, reduction, wn=wn, bias=True,
                bn=False, act=nn.ReLU(True), res_scale=res_scale, da=da) \
            for _ in range(n_resblocks)]
        modules_body.append(
            PyConv2d(in_channels=n_feat,
                out_channels=[n_feat//4, n_feat//4, n_feat//2],
                pyconv_kernels=[3, 5, 7],
                pyconv_groups=[1, 4, 8]))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class WideActResBlock(nn.Module):
    def __init__(self, nf=64):
        super(WideActResBlock, self).__init__()
        self.res_scale = 1
        body = []
        expand = 6
        linear = 0.8
        kernel_size = 3
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act=nn.ReLU(True)

        body.append(
            wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class PSWideActResBlock(nn.Module):
    def __init__(self, nf=64):
        super(PSWideActResBlock, self).__init__()
        self.res_scale = 1
        body = []
        expand = 6
        linear = 0.75
        kernel_size = 3
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act=nn.ReLU(True)

        body.append(
            wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
        body.append(
            wn(PSConv2d(int(nf*linear), nf, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class PyWideActResBlock(nn.Module):
    def __init__(self, nf=64):
        super(PyWideActResBlock, self).__init__()
        self.res_scale = 1
        body = []
        expand = 6
        linear = 0.75
        kernel_size = 3
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act=nn.ReLU(True)
        expand_nf = nf*expand
        linear_nf = int(nf * linear)

        body.append(
            wn(nn.Conv2d(nf, nf*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(nf*expand, int(nf*linear), 1, padding=1//2)))
        body.append(
            PyConv2d(in_channels=linear_nf,
                out_channels=[nf//4, nf//4, nf//2],
                pyconv_kernels=[3, 5, 7],
                pyconv_groups=[1, 4, 8]))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    x = x.float()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output


# def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
#     """Warp an image or feature map with optical flow
#     Args:
#         x (Tensor): size (N, C, H, W)
#         flow (Tensor): size (N, H, W, 2), normal value
#         interp_mode (str): 'nearest' or 'bilinear'
#         padding_mode (str): 'zeros' or 'border' or 'reflection'

#     Returns:
#         Tensor: warped image or feature map
#     """
#     assert x.size()[-2:] == flow.size()[1:3]
#     B, C, H, W = x.size()
#     # mesh grid
#     grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
#     grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
#     grid.requires_grad = False
#     grid = grid.type_as(x)
#     vgrid = grid + flow
#     # scale grid to [-1,1]
#     vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
#     vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
#     vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
#     output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
#     return output
