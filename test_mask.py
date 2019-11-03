"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt 

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import get_config
from trainer import Trainer

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml')
parser.add_argument('--ckpt',
                    type=str,
                    default='pretrained/animal149_gen.pt')
parser.add_argument('--input_image',
                    type=str,
                    default='images/input_content.jpg')
parser.add_argument('--output_folder', 
                    type=str, 
                    default='images/')
parser.add_argument('--num_plot', 
                    type=int, 
                    default=5)
opts = parser.parse_args()
cudnn.benchmark = True
opts.vis = True
config = get_config(opts.config)
config['batch_size'] = 1
config['gpus'] = 1

trainer = Trainer(config)
trainer.cuda()
trainer.load_ckpt(opts.ckpt)
trainer.eval()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = [transforms.Resize((128, 128))] + transform_list
transform = transforms.Compose(transform_list)

image = Image.open(opts.input_image).convert('RGB')
img = transform(image).unsqueeze(0)

with torch.no_grad():
    content, style = trainer.debug(img)

content = content.squeeze(0).cpu().numpy()
style = style.squeeze(0).cpu().numpy()
index_c = np.arange(content.shape[0])
index_s = np.arange(style.shape[0])

print('content shape:')
print(content.shape)
print('style shape:')
print(style.shape)

plt.figure()
# line 1: input
plt.subplot(3, opts.num_plot, (opts.num_plot+1)//2)
plt.imshow(image)
plt.axis('off')
plt.title('input')
# line 2: content
base = opts.num_plot+1
index_c = np.random.permutation(index_c)
for i in range(opts.num_plot):
    k = index_c[i]
    plt.subplot(3, opts.num_plot, base+i)
    c = content[k, :, :]
    plt.imshow(c)
    plt.axis('off')
    plt.title('content-%d' % k)
# line 3: style
base += opts.num_plot
index_s = np.random.permutation(index_s)
for i in range(opts.num_plot):
    k = index_s[i]
    plt.subplot(3, opts.num_plot, base+i)
    s = style[k, :, :]
    plt.imshow(s)
    plt.axis('off')
    plt.title('style-%d' % k)

plt.savefig(os.path.join(opts.output_folder, 'latent.jpg'))

zero_style = 0
for i in range(style.shape[0]):
    s = style[i, :, :]
    if np.sum(s) == 0:
        zero_style += 1
print('number of zero styles (%d in all):' % style.shape[0])
print(zero_style)
