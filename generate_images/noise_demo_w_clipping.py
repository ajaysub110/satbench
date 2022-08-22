import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# load image and set hyperparameters
noise_sd = 0.12 # SD of 0.1 keeps all pixels between 0 and 1
image_contrast = 0.1
imagenet_mean = 0.449
cbfs = [10 * 2**i for i in range(0, 4)]
print(cbfs)

fig, ax = plt.subplots(1, len(cbfs))
for i, cbf in enumerate(cbfs):
	bw = 0.7 * cbf
	lpf_r1 = int(cbf - bw/2)
	lpf_r2 = int(cbf + bw/2)

	print(lpf_r1, lpf_r2)

	image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/6_color_gray_g_knife.JPEG').convert('L'), dtype=np.float32)
	# image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/34_color_gray_g_elephant.JPEG').convert('L'), dtype=np.float32)
	# image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/86_color_gray_g_chair.JPEG').convert('L'), dtype=np.float32)

	# normalize image to 0-1 with imagenet mean
	image = (image / 255.0)
	image = (image - image.mean()) * image_contrast + imagenet_mean

	# print(image.mean())
	# plt.hist(image.ravel(), bins=100)
	# plt.show()

	# create noise array of same size as image with 0 mean
	noise = np.random.randn(*image.shape) * noise_sd

	# find fft of noise
	Fnoise = np.fft.fftshift(np.fft.fft2(noise))

	# Create a low pass filter mask
	x, y = noise.shape[0], noise.shape[1]
	bbox1 = ((x/2)-(lpf_r1/2),(y/2)-(lpf_r1/2),(x/2)+(lpf_r1/2),(y/2)+(lpf_r1/2))
	bbox2 = ((x/2)-(lpf_r2/2),(y/2)-(lpf_r2/2),(x/2)+(lpf_r2/2),(y/2)+(lpf_r2/2))
	lpf = Image.new("L", (noise.shape[0], noise.shape[1]), color=0)
	lpf_draw = ImageDraw.Draw(lpf)
	lpf_draw.ellipse(bbox2, fill=1)
	lpf_draw.ellipse(bbox1, fill=0)
	lpf_np = np.array(lpf, dtype=np.float32)

	# Filter noise using low pass filter
	Fnoise_filtered = np.multiply(Fnoise, lpf_np)

	# Inverse FFT
	noise_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(Fnoise_filtered)))

	# print(noise_filtered.mean())
	# plt.hist(noise_filtered.ravel(), bins=100)
	# plt.show()

	# Add filtered noise to image
	noisy_image = image + noise_filtered
	if np.any(noisy_image < 0) or np.any(noisy_image > 1):
		print("WARNING: Image contains negative or >1 values")
	# print(noisy_image.mean())
	# plt.hist(noisy_image.ravel(), bins=100)
	# plt.show()

	ax[i].imshow(noisy_image, cmap='gray')
	ax[i].set_axis_off()
	ax[i].set_title("r1: {}, r2: {}".format(lpf_r1, lpf_r2))
	# fig.savefig('noise_filtered_r1={}_r2={}.png'.format(lpf_r1, lpf_r2))
	
fig.tight_layout()
plt.show()
