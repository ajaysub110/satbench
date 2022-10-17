import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

import pyrtools as pt

order = 0
imsize = 224
noise_sds = [0.001, 0.02, 0.04, 0.06, 0.08, 0.1]
imagenet_mean = 0.449
image_contrast = 0.2

root = '/Users/ajay/code/anytime-prediction-data/GraySplit/0'
filenames = os.listdir(root)
images = []
n_freqs = 7
n_noises = len(noise_sds)
randnums = random.sample(range(0,len(filenames)), n_freqs * n_noises)

def rmspower(im):
	return np.sqrt(np.mean(np.square(im)))

fig, ax = plt.subplots(n_noises, n_freqs)
for i in range(n_noises):
	for j in range(n_freqs):
		noise_sd = noise_sds[i]
		filename = filenames[randnums[n_freqs*i + j]]
		image = np.array(Image.open(os.path.join(root, filename)).convert('L'), dtype=np.float32)

		noise = np.random.randn(*image.shape) * noise_sd

		pyr = pt.pyramids.LaplacianPyramid(noise)

		recons = []
		for bandi in range(len(pyr.pyr_coeffs.keys())):
			recons.append(pyr.recon_pyr(levels=bandi))

		for ri in range(1, len(recons)):
			recons[ri] = recons[ri] * rmspower(recons[0]) / rmspower(recons[ri])

		# normalize image to 0-1 with imagenet mean
		image = (image / 255.0)
		image = (image - image.mean()) * image_contrast + imagenet_mean

		noisyims = [image + n for n in recons]
		# print(len(noisyims))

		noisyim = noisyims[-(j+1)]
		# print(noisyim.shape)
		ax[i,j].imshow(noisyim, cmap='gray', vmin=0, vmax=1)

for a in ax:
	for b in a:
		b.set_xticklabels([])
		b.set_yticklabels([])
		b.spines['top'].set_visible(False)
		b.spines['right'].set_visible(False)
		b.spines['bottom'].set_visible(False)
		b.spines['left'].set_visible(False)
		b.get_xaxis().set_ticks([])
		b.get_yaxis().set_ticks([])
		b.set_aspect('equal')

fig.subplots_adjust(wspace=0, hspace=0)
plt.show()