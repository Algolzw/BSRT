import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.arch_util as arch_util
from torch.cuda.amp import autocast
import model.swin_util as swu
import time
import os
import math
from utils.debayer import Debayer3x3
import torchvision.utils as tvutils
from datasets.burstsr_dataset import pack_raw_image, flatten_raw_image_batch

try:
    from model.non_local.non_local_cross_dot_product import NONLocalBlock2D as NonLocalCross
    from model.non_local.non_local_dot_product import NONLocalBlock2D as NonLocal
except ImportError:
    raise ImportError('Failed to import Non_Local module.')

try:
    from model.DCNv2.dcn_v2 import DCN_sep as DCN, FlowGuidedDCN, InsideFlowGuidedDCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


def make_model(args, parent=False):
    nframes = args.burst_size
    img_size = args.patch_size * 2
    patch_size = 1
    in_chans = args.burst_channel
    out_chans = args.n_colors
    
    if args.model_level == "S":
        depths = [6]*1 + [6] * 4
        num_heads = [6]*1 + [6] * 4
        embed_dim = 60
    elif args.model_level == "L":
        depths = [6]*1 + [8] * 6
        num_heads = [6]*1 + [6] * 6
        embed_dim = 180
    window_size = 8
    mlp_ratio = 2
    upscale = args.scale[0]
    non_local = args.non_local
    use_checkpoint=args.use_checkpoint

    if args.local_rank <= 0:
        print("depths: ", depths)

    return BSRT(args=args,nframes=nframes,
                   img_size=img_size,
                   patch_size=patch_size,
                   in_chans=in_chans,
                   out_chans=out_chans,
                   embed_dim=embed_dim,
                   depths=depths,
                   num_heads=num_heads,
                   window_size=window_size,
                   mlp_ratio=mlp_ratio,
                   upscale=upscale,
                   non_local=non_local,
                   use_checkpoint=use_checkpoint)


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        # ref = [ref]
        # supp = [supp]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                arch_util.flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2**(5-level) # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h//scale, w//scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w//scale) / float(w_floor//scale)
                flow_out[:, 1, :, :] *= float(h//scale) / float(h_floor//scale)
                if torch.abs(flow_out).mean() > 200:
                    print(f"level {level}, flow > 200: {torch.abs(flow_out).mean():.4f}")
                    # return None
                    flow_out.clamp(-250, 250)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list



class FlowGuidedPCDAlign(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    '''

    def __init__(self, nf=64, groups=8):
        super(FlowGuidedPCDAlign, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, nbr_fea_warped_l, ref_fea_l, flows_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_warped_l[2], ref_fea_l[2], flows_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset, flows_l[2]))
        # L2
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = torch.cat([nbr_fea_warped_l[1], ref_fea_l[1], flows_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset, flows_l[1])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = torch.cat([nbr_fea_warped_l[0], ref_fea_l[0], flows_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset, flows_l[0])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.cas_dcnpack(L1_fea, offset)

        return L1_fea


class CrossNonLocal_Fusion(nn.Module):
    ''' Cross Non Local fusion module
    '''
    def __init__(self, nf=64, out_feat=96, nframes=5, center=2):
        super(CrossNonLocal_Fusion, self).__init__()
        self.center = center

        self.non_local_T = nn.ModuleList()
        self.non_local_F = nn.ModuleList()

        for i in range(nframes):
            self.non_local_T.append(NonLocalCross(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))
            self.non_local_F.append(NonLocal(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf*2, out_feat, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        ref = aligned_fea[:, self.center, :, :, :].clone()

        cor_l = []
        non_l = []
        for i in range(N):
            nbr = aligned_fea[:, i, :, :, :]
            non_l.append(self.non_local_F[i](nbr))
            cor_l.append(self.non_local_T[i](nbr, ref))

        aligned_fea_T = torch.cat(cor_l, dim=1)
        aligned_fea_F = torch.cat(non_l, dim=1)
        aligned_fea = torch.cat([aligned_fea_T, aligned_fea_F], dim=1)

        #### fusion
        fea = self.fea_fusion(aligned_fea)

        return fea



class BSRT(nn.Module):
    def __init__(self, args, nframes=8, img_size=64, patch_size=1, in_chans=3, out_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, non_local=False,
                 **kwargs):
        super(BSRT, self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        num_feat = 64
        groups = 8
        # embed_dim = num_feat
        back_RBs = 5
        n_resblocks = 6

        self.args = args
        self.center = 0
        self.upscale = upscale
        self.window_size = window_size
        self.non_local = non_local
        self.nframes = nframes

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        spynet_path='/home/luoziwei/.pretrained_models/spynet_sintel_final-3d2a1287.pth'
        self.spynet = SpyNet(spynet_path, [3, 4, 5])
        self.conv_flow = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.flow_ps = nn.PixelShuffle(2)

        # split image into non-overlapping patches
        self.patch_embed = swu.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = swu.PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch*(1+2*0), embed_dim, 3, 1, 1, bias=True)
        
        # # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        if args.swinfeature:
            if self.args.local_rank <= 0:
                print("using swinfeature")
            self.pre_layers = nn.ModuleList()
            for i_layer in range(depths[0]):
                layer = swu.SwinTransformerBlock(dim=embed_dim, 
                            input_resolution=(patches_resolution[0]//2,
                                              patches_resolution[1]//2),
                             num_heads=num_heads[0], window_size=window_size,
                             shift_size=0 if (i_layer % 2 == 0) else window_size // 2,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[i_layer],
                             norm_layer=norm_layer)
                self.pre_layers.append(layer)

            self.pre_norm = norm_layer(embed_dim)
        else:
            WARB = functools.partial(arch_util.WideActResBlock, nf=embed_dim)
            self.feature_extraction = arch_util.make_layer(WARB, 5)

        self.conv_after_pre_layer = nn.Conv2d(embed_dim, num_feat*4, 3, 1, 1, bias=True)
        self.mid_ps = nn.PixelShuffle(2)

        self.fea_L2_conv1 = nn.Conv2d(num_feat, num_feat*2, 3, 2, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(num_feat*2, num_feat*4, 3, 2, 1, bias=True)

        #####################################################################################################
        ################################### 2, Feature Enhanced PCD Align ###################################

        # Top layers
        self.toplayer = nn.Conv2d(num_feat*4, num_feat, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(num_feat*2, num_feat, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(num_feat*1, num_feat, kernel_size=1, stride=1, padding=0)

        # self.align = PCD_Align(nf=num_feat, groups=groups)
        self.align = FlowGuidedPCDAlign(nf=num_feat, groups=groups)
        #####################################################################################################
        ################################### 3, Multi-frame Feature Fusion  ##################################

        if self.non_local:
            if self.args.local_rank <= 0:
                print("using non_local")
            self.fusion = CrossNonLocal_Fusion(nf=num_feat, out_feat=embed_dim, nframes=nframes, center=self.center)
        else:
            self.fusion = nn.Conv2d(nframes * num_feat, embed_dim, 1, 1, bias=True)

        #####################################################################################################
        ################################### 4, deep feature extraction ######################################

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            swu.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(1, self.num_layers):
            layer = swu.RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size
                         )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################ 5, high quality image reconstruction ################################

        self.upconv1 = nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, args.n_colors, 3, 1, 1, bias=True)

        #### skip #############
        self.skip_pixel_shuffle = nn.PixelShuffle(2)
        self.skipup1 = nn.Conv2d(num_in_ch//4, num_feat * 4, 3, 1, 1, bias=True)
        self.skipup2 = nn.Conv2d(num_feat, args.n_colors * 4, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            swu.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def pre_forward_features(self, x):
        if self.args.swinfeature:
            x_size = (x.shape[-2], x.shape[-1])
            x = self.patch_embed(x, use_norm=True)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

            for idx, layer in enumerate(self.pre_layers):
                x = layer(x, x_size)

            x = self.pre_norm(x)
            x = self.patch_unembed(x, x_size)

        else:
            x = self.feature_extraction(x)

        return x

    def forward_features(self, x):
        x_size = (x.shape[-2], x.shape[-1])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, x_size)
            if torch.any(torch.isinf(x)) or torch.any(torch.isnan(x)):
                print('layer: ', idx)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    @autocast()
    def forward(self, x, print_time=False):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### skip module ########
        skip1 = self.lrelu2(self.skip_pixel_shuffle(self.skipup1(self.skip_pixel_shuffle(x_center))))
        skip2 = self.skip_pixel_shuffle(self.skipup2(skip1))

        x_ = self.conv_flow(self.flow_ps(x.view(B*N, C, H, W))).view(B, N, -1, H*2, W*2)
        
        # calculate flows
        ref_flows = self.get_ref_flows(x_)

        #### extract LR features
        x = self.lrelu(self.conv_first(x.view(B*N, -1, H, W)))

        L1_fea = self.mid_ps(self.conv_after_pre_layer(self.pre_forward_features(x)))
        _, _, H, W = L1_fea.size()

        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))

        # FPN enhance features
        L3_fea = self.lrelu(self.toplayer(L3_fea))
        L2_fea = self.smooth1(self._upsample_add(L3_fea, self.latlayer1(L2_fea)))
        L1_fea = self.smooth2(self._upsample_add(L2_fea, self.latlayer2(L1_fea)))

        L1_fea = L1_fea.view(B, N, -1, H, W).contiguous()
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2 ).contiguous()
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4).contiguous()

        #### PCD align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), 
            L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), 
                L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            flows_l = [
                ref_flows[0][:, i, :, :, :].clone(), 
                ref_flows[1][:, i, :, :, :].clone(), 
                ref_flows[2][:, i, :, :, :].clone()
            ]
            # print(nbr_fea_l[0].shape, flows_l[0].shape)
            nbr_warped_l = [
                arch_util.flow_warp(nbr_fea_l[0], flows_l[0].permute(0, 2, 3, 1), 'bilinear'),
                arch_util.flow_warp(nbr_fea_l[1], flows_l[1].permute(0, 2, 3, 1), 'bilinear'),
                arch_util.flow_warp(nbr_fea_l[2], flows_l[2].permute(0, 2, 3, 1), 'bilinear')
            ]
            aligned_fea.append(self.align(nbr_fea_l, nbr_warped_l, ref_fea_l, flows_l))

        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]

        if not self.non_local:
            aligned_fea = aligned_fea.view(B, -1, H, W)

        x = self.lrelu(self.fusion(aligned_fea))

        x = self.lrelu(self.conv_after_body(self.forward_features(x))) + x

        x = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
        x = skip1 + x
        x = self.lrelu(self.pixel_shuffle(self.upconv2(x)))
        x = self.lrelu(self.HRconv(x))
        x = self.conv_last(x)

        x = skip2 + x
        return x


    def get_ref_flows(self, x):
        '''Get flow between frames ref and other'''

        b, n, c, h, w = x.size()
        x_nbr = x.reshape(-1, c, h, w)
        x_ref = x[:, self.center:self.center+1, :, :, :].repeat(1, n, 1, 1, 1).reshape(-1, c, h, w)

        # backward
        flows = self.spynet(x_ref, x_nbr)
        flows_list = [flow.view(b, n, 2, h // (2 ** (i)), w // (2 ** (i))) for flow, i in
                          zip(flows, range(3))]

        return flows_list







