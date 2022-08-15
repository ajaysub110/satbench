import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

image = np.array(Image.open('/Users/ajay/code/anytime-prediction-data/ColorGraySplit/0/6_color_gray_g_knife.JPEG').convert('L'), dtype=np.float32) / 255.
noise_sd = 0.1 # SD of 0.1 keeps all pixels between 0 and 1
lpf_r1, lpf_r2 = 0, 320
# lpf_r1, lpf_r2 = 80, 160
# lpf_r1, lpf_r2 = 160, 240
# lpf_r1, lpf_r2 = 240, 320

# create noise array of same size as image
noise = np.random.randn(*image.shape) * noise_sd + 0.5

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
noise_filtered = np.maximum(0, np.minimum(noise_filtered, 1))

print(np.max(noise_filtered), np.min(noise_filtered))

# Add filtered noise to image
print(np.max(image), np.min(image))
noisy_image = image + noise_filtered
print(np.max(noisy_image), np.min(noisy_image))
plt.hist(noisy_image.ravel(), bins=100)

# fig = plt.figure()
# img = plt.imshow(noisy_image, cmap='gray')
plt.show()
# fig.savefig('noise_filtered_r1={}_r2={}.png'.format(lpf_r1, lpf_r2))
