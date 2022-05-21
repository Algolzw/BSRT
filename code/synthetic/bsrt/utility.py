import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

import numpy as np
import imageio
import os
import sys

import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import torch.distributed as dist
import matplotlib

matplotlib.use('Agg')


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdims=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdims=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * torch.abs(dx) / 2.0
    loss_y = weights_y * torch.abs(dy) / 2.0
    return torch.mean(loss_x) / 2.0 + torch.mean(loss_y) / 2.0

def smooth_loss(flow, img):
    loss = smooth_grad_1st(flow, img, 10)
    return sum([torch.mean(loss)])
    

def setup(rank, world_size):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "tcp://localhost:1234"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '4321'
        # if mp.get_start_method(allow_none=True) is None:
        if (
            mp.get_start_method(allow_none=True) != "spawn"
        ):  # Return the name of start method used for starting processes
            mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        # for d in args.data_test:
        #     os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


def write_gray_to_tfboard(img):
    img_debug = img[0, ...].detach().cpu().numpy()

    # img_debug = cv2.normalize(img_debug, None, 0, 255,
    #                           cv2.NORM_MINMAX, cv2.CV_8U)
    img_debug = img_debug * 255
    img_debug = np.clip(img_debug, 0, 255)
    img_debug = img_debug.astype(np.uint8)
    return img_debug[0, ...]





######################## BayerUnifyAug ############################

BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
NORMALIZATION_MODE = ["crop", "pad"]


def bayer_unify(raw, input_pattern, target_pattern, mode) -> np.ndarray:
    """
    Convert a bayer raw image from one bayer pattern to another.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """

    if input_pattern == target_pattern:
        h_offset, w_offset = 0, 0
    elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
        h_offset, w_offset = 1, 0
    elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
        h_offset, w_offset = 0, 1
    elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError('Unexpected pair of input and target bayer pattern!')

    if mode == "pad":
        # out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
        out = F.pad(raw, (w_offset, w_offset, h_offset, h_offset), mode='reflect')
    elif mode == "crop":
        _, _, _, h, w = raw.shape
        out = raw[..., h_offset:h - h_offset, w_offset:w - w_offset]
    else:
        raise ValueError('Unknown normalization mode!')

    return out


def bayer_aug(raw, flip_h=False, flip_w=False, transpose=False, input_pattern='RGGB') -> np.ndarray:
    """
    Apply augmentation to a bayer raw image.
    """

    aug_pattern, target_pattern = input_pattern, input_pattern

    out = raw
    if flip_h:
        out = torch.flip(out, [3]) # GBRG, RGGB
        aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
    if flip_w:
        out = torch.flip(out, [4])
        aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
    if transpose:
        out = out.permute(0, 1, 2, 4, 3)
        aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]

    out = bayer_unify(out, aug_pattern, target_pattern, "crop")
    return out












