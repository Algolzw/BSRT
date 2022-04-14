import numpy as np
import torch
import cv2 as cv


def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)


def torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).cpu().numpy()


def torch_to_npimage(a: torch.Tensor, unnormalize=True):
    a_np = torch_to_numpy(a)

    if unnormalize:
        a_np = a_np * 255
    a_np = a_np.astype(np.uint8)
    return cv.cvtColor(a_np, cv.COLOR_RGB2BGR)


def npimage_to_torch(a, normalize=True, input_bgr=True):
    if input_bgr:
        a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
    a_t = numpy_to_torch(a)

    if normalize:
        a_t = a_t / 255.0

    return a_t


def convert_dict(base_dict, batch_sz):
    out_dict = []
    for b_elem in range(batch_sz):
        b_info = {}
        for k, v in base_dict.items():
            if isinstance(v, (list, torch.Tensor)):
                b_info[k] = v[b_elem]
        out_dict.append(b_info)

    return out_dict