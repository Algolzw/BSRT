import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
import time

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        self.args = args
        if args.local_rank == 0:
            print("Making model: ", args.model)
            print("Patch size: ", args.patch_size)


        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda:%d' % args.local_rank)
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

        # time.sleep(3)

        if args.n_GPUs > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank],
                find_unused_parameters=True
                )

        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            # if self.n_GPUs > 1:
            return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                # return self.model(x)
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )
        if self.n_GPUs > 1:
            model = self.model.module
        else:
            model = self.model

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                if self.args.local_rank == 0:
                    print('Load the model from {}'.format(pre_train))
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.local_rank}
                load_from = torch.load(pre_train, map_location=map_location)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=True)
            del load_from

        
        if self.args.finetune:
            if self.args.local_rank == 0:
                print('finetune')
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.HRconv.parameters():
                param.requires_grad = True
            for param in self.model.conv_last.parameters():
                param.requires_grad = True

        if self.args.finetune_prelayer:
            if self.args.local_rank == 0:
                print('finetune_prelayer')
            if self.args.swinfeature:
                if self.args.model == 'MBSRT':
                    for param in self.model.pre_layer1.parameters():
                        param.requires_grad = True
                    for param in self.model.pre_layer2.parameters():
                        param.requires_grad = True
                else:
                    for param in self.model.pre_layers.parameters():
                        param.requires_grad = True
            else:
                for param in self.model.feature_extraction.parameters():
                    param.requires_grad = True

            for param in self.model.conv_after_pre_layer.parameters():
                param.requires_grad = True

        if self.args.finetune_align:
            if self.args.local_rank == 0:
                print('finetune_align')
            for param in self.model.align.parameters():
                param.requires_grad = True

        if self.args.finetune_spynet:
            if self.args.local_rank == 0:
                print('finetune_spynet')
            for param in self.model.spynet.parameters():
                param.requires_grad = True

        if self.args.finetune_swin:
            if self.args.local_rank == 0:
                print('finetune_swin')
            for param in self.model.layers.parameters():
                param.requires_grad = True
            for param in self.model.conv_after_body.parameters():
                param.requires_grad = True

        if self.args.finetune_upconv:
            if self.args.local_rank == 0:
                print('finetune_upconv')
            for param in self.model.upconv1.parameters():
                param.requires_grad = True
            for param in self.model.upconv2.parameters():
                param.requires_grad = True
            for param in self.model.skipup1.parameters():
                param.requires_grad = True
            for param in self.model.skipup2.parameters():
                param.requires_grad = True

        if self.args.finetune_conv:
            if self.args.local_rank == 0:
                print('finetune_conv')
            # for param in self.model.conv_first.parameters():
            #     param.requires_grad = True
            # for param in self.model.conv_flow.parameters():
            #     param.requires_grad = True
            # for param in self.model.fea_L2_conv1.parameters():
            #     param.requires_grad = True
            # for param in self.model.fea_L3_conv1.parameters():
            #     param.requires_grad = True
            # for param in self.model.toplayer.parameters():
            #     param.requires_grad = True
            # for param in self.model.smooth1.parameters():
            #     param.requires_grad = True
            # for param in self.model.smooth2.parameters():
            #     param.requires_grad = True
            # for param in self.model.latlayer1.parameters():
            #     param.requires_grad = True
            # for param in self.model.latlayer2.parameters():
            #     param.requires_grad = True
            # for param in self.model.fusion.parameters():
            #     param.requires_grad = True
            for param in self.model.conv_after_body.parameters():
                param.requires_grad = True
            
            

    def forward_chop(self, *args, shave=10, min_size=160000):
        scale = 1 if self.input_large else self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
