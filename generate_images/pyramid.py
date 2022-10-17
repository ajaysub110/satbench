import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pyrtools as pt

order = 0
imsize = 224
noise_sd = 0.1
imagenet_mean = 0.449
image_contrast = 0.1

image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/6_color_gray_g_knife.JPEG').convert('L'), dtype=np.float32)
# image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/34_color_gray_g_elephant.JPEG').convert('L'), dtype=np.float32)
# image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/86_color_gray_g_chair.JPEG').convert('L'), dtype=np.float32)

def rmspower(im):
	return np.sqrt(np.mean(np.square(im)))

noise = np.random.randn(*image.shape) * noise_sd

pyr = pt.pyramids.LaplacianPyramid(noise)

recons = []
for bandi in range(len(pyr.pyr_coeffs.keys())):
	recons.append(pyr.recon_pyr(levels=bandi))


for i in range(1, len(recons)):
	recons[i] = recons[i] * rmspower(recons[0]) / rmspower(recons[i])
	print(rmspower(recons[i]))
print(len(recons), recons[-1].shape)

# pt.imshow([recon + imagenet_mean for recon in recons], zoom = 0.5)
# plt.show()

allrecon = pyr.recon_pyr(levels='all')
# pt.imshow(allrecon)
# plt.show()

pt.image_compare(noise, allrecon)

# normalize image to 0-1 with imagenet mean
image = (image / 255.0)
image = (image - image.mean()) * image_contrast + imagenet_mean

noisyims = [image + n for n in recons]

pt.imshow(noisyims, zoom = 0.5, vrange=(0,1))
plt.show()