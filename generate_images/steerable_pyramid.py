import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import pyrtools as pt
import plenoptic as po
from plenoptic.simulate import Steerable_Pyramid_Freq 
from plenoptic.tools.data import to_numpy 
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
DATA_PATH = os.path.abspath('../data')

order = 0
imsize = 224
noise_sd = 0.12
imagenet_mean = 0.449
image_contrast = 0.1
pyr = Steerable_Pyramid_Freq(height=5,image_shape=[imsize,imsize],order=order,is_complex = False,twidth=1, downsample=True)
pyr.to(device)

# empty_image = torch.zeros((1,1,imsize,imsize),dtype=dtype).to(device)
empty_image = torch.randn((1,1,imsize,imsize), dtype=dtype).to(device) * noise_sd
pyr_coeffs = pyr.forward(empty_image)

for k,v in pyr.pyr_size.items():
    mid = (v[0]//2, v[1]//2)
    pyr_coeffs[k][0,0,mid[0],mid[1]]=1

reconList = []
for k in pyr_coeffs.keys():
    if isinstance(k, tuple):
        reconList.append(pyr.recon_pyr(pyr_coeffs, k[0], k[1]))
    else:
        reconList.append(pyr.recon_pyr(pyr_coeffs, k))

# po.imshow(reconList, col_wrap=order+1, vrange='indep1', zoom=1)
print(len(reconList))
reconListsamp = [empty_image]
for i in range(1, len(reconList)-1, 2):
	reconListsamp.append((reconList[i] + reconList[i+1])/2)

# po.imshow(reconListsamp, col_wrap=order+1, vrange='indep1', zoom=2)

def imshow(l):
	fig, ax = plt.subplots(1, len(l))
	for i, im in enumerate(l):
		if type(im) != np.ndarray:
			im = im.to('cpu').numpy().reshape((imsize, imsize))

		# run range check
		# print(im.min(), im.max())

		ax[i].imshow(im, cmap='gray')
		ax[i].axis('off')

# imshow(reconListsamp)
plt.show()

avg_noise = sum(reconListsamp) / len(reconListsamp)

# print(avg_noise.min().item(), avg_noise.max().item())
# plt.imshow(avg_noise.reshape((imsize, imsize)), cmap='gray')
plt.show()

image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/6_color_gray_g_knife.JPEG').convert('L'), dtype=np.float32)
# image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/34_color_gray_g_elephant.JPEG').convert('L'), dtype=np.float32)
# image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/86_color_gray_g_chair.JPEG').convert('L'), dtype=np.float32)

# normalize image to 0-1 with imagenet mean
image = (image / 255.0)
image = (image - image.mean()) * image_contrast + imagenet_mean

print(image.min(), image.max())

noiseList = [im.to('cpu').numpy().reshape((imsize, imsize)) for im in reconListsamp]

print([(n.min(), n.max()) for n in noiseList])

noiseImageList = [image + n for n in noiseList]

imshow(noiseImageList)
plt.tight_layout()
plt.show()