from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import cv2
import numpy as np
import torch

from models.decode import _topk
from models.model import create_model, load_model
from opts import opts
from utils.image import get_affine_transform

image_ext = ['jpg', 'jpeg', 'png', 'webp']


def pre_process(image, scale=1):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    if opt.fix_res:
        inp_height, inp_width = opt.input_h, opt.input_w
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
    else:
        inp_height = (new_height | opt.pad) + 1
        inp_width = (new_width | opt.pad) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)) /
                 np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    return images


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt.device = torch.device('cuda')

    print('Loading images...')
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)

    for (image_name) in image_names:
        image = cv2.imread(image_name)
        original_images = pre_process(image)
        original_images = original_images.to(opt.device)
        images = torch.tensor(original_images)
        images.requires_grad = True
        for i in range(20):
            hm = model(images)[-1]['hm'].sigmoid_()
            scores = _topk(hm, K=1)
            loss = torch.sum(scores)
            if loss > 0:
                print(loss)
                model.zero_grad()
                loss.backward()
                grad = images.grad.data.sign()
                images = images - 0.4 * grad
                images = torch.clamp(images, -1, 1)
            else:
                break
        perturb = (images - original_images).squeeze(0).numpy()
        perturb = perturb.transpose(1, 2, 0)
        perturb = perturb * np.array(opt.std, dtype=np.float32).reshape(1, 1, 3).astype(np.float32) * 255
        perturb = (perturb.astype(np.int16) + 128).clip(0, 255)
        perturb = cv2.resize(perturb, (image.shape[0:2]))
        perturb = perturb - 128
        adv_image = (image + perturb).clip(0, 255)
        cv2.imwrite('results/' + image_name, adv_image)


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
