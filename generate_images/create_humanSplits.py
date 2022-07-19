import os
import argparse
import random
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from ImagenetCOCOMapping import mappings

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/Users/ajay/code/anytime-prediction-data/16ClassImagenet')
parser.add_argument('--output_dir', type=str, default='/Users/ajay/code/anytime-prediction-data/NoiseSplit_gray_contrast0.2')
parser.add_argument('--n_modes', type=int, default=3) # number of noise, blur or color values.
parser.add_argument('--n_rts', type=int, default=5) # number of reaction time blocks

parser.add_argument('--mode', type=str, default='noise')
args = parser.parse_args()
args.dataset_dir = '/Users/ajay/Desktop/Datasets/Imagenet_ILSVRC2012_classification-localization/ILSVRC/Data/CLS-LOC/val'

n_cat_training = 50
n_time_training = 50
n_samples_needed_per_mode = (1000 + n_time_training) // (args.n_rts * args.n_modes) # 105

class AddGaussianBlur(object):
    def __init__(self, kernel=7, std=1.0):
        self.kernel = kernel
        self.std = std
    
    def __call__(self, tensor):
        if self.std != 0.0:
            tensor = transforms.GaussianBlur(kernel_size = 7,sigma=self.std)(tensor)

        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoise(object):
    """
    Author: Omkar Kumbhar
    Description:
    Adding gaussian noise to images in the batch
    """
    def __init__(self, mean=0., std=1., contrast=0.1):
        self.std = std
        self.mean = mean
        self.contrast = contrast

    def __call__(self, tensor):
        noise = torch.Tensor()
        n = tensor.size(1) * tensor.size(2)
        sd2 = self.std * 2

        while len(noise) < n:
            # more samples than we require
            m = 2 * (n - len(noise))
            new = torch.randn(m) * self.std

            # remove out-of-range samples
            new = new[new >= -sd2]
            new = new[new <= sd2]

            # append to noise tensor
            noise = torch.cat([noise, new])
        
        # pick first n samples and reshape to 2D
        noise = torch.reshape(noise[:n], (tensor.size(1), tensor.size(2)))

        # stack noise and translate by mean to produce std + 
        newnoise = torch.stack([noise, noise, noise]) + self.mean

        # shift image hist to mean = 0.5
        tensor = tensor + (0.5 - tensor.mean())

        # self.contrast = 1.0 / (5. * max(1.0, tensor.max() + sd2, 1.0 + (0 - tensor.min() - sd2)))
        # print(self.contrast)

        tensor = transforms.functional.adjust_contrast(tensor, self.contrast)
        
        return tensor + newnoise + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def create_splits_flist():
	def get_fcount(category):
		return len(os.listdir(os.path.join(args.data_dir, category)))

	def cat(l, c):
		return [(x, c) for x in l]
	fcounts = dict([(category, get_fcount(category)) for category in mappings.keys()]) # {'knife': 50, 'chair': 50, ...}
	
	# initialize flist
	flist = {}
	flist['train'] = []
	# create keys for each rt
	for rt in range(args.n_rts):
		flist[str(rt)] = []

	# for each category, sample required images
	n_samples_needed_per_category = n_samples_needed_per_mode * args.n_modes * args.n_rts // len(mappings.keys()) # 65
	n_sampled_earlier = dict([(category, 0) for category in mappings.keys()]) # 0

	# split files into rt lists
	remaining_files = []
	remaining_cat_counts = {}
	for category in mappings.keys():
		n_sampleable = n_samples_needed_per_category if fcounts[category] > n_samples_needed_per_category else fcounts[category] # 50, 65
		n_sampleable_per_rt = int(n_sampleable / args.n_rts) # 10, 13
		# n_sampleable_per_rt = n_sampleable * args.n_modes // (args.n_modes * args.n_rts + 1) # 9
		files = os.listdir(os.path.join(args.data_dir, category))

		for rt in range(args.n_rts):
			# add 9 files to each rt
			flist[str(rt)] += cat(files[n_sampled_earlier[category]:n_sampled_earlier[category] + n_sampleable_per_rt], category)
			n_sampled_earlier[category] += n_sampleable_per_rt # 10, 13

		# n_sampled_earlier[category] = 50, 65
		remaining_cat_counts[category] = len(files[n_sampled_earlier[category]:])
		remaining_files += cat(files[n_sampled_earlier[category]:], category) # 50 - 50 = 0, 100 - 65 = 35


	# tally: flist[rt] = 160 or 208, remaining_files = 0 or 560 (all from categories that don't have 50)

	# fill up each rt from remaining files
	for rt in range(args.n_rts):
		flist[str(rt)] += random.sample(remaining_files, args.n_modes * n_samples_needed_per_mode - len(flist[str(rt)])) # 210 - 160 (if 50) or 208 (if 65) = 50 or 2

	# pick 50 randomly for training
	print(remaining_cat_counts)
	nonzero_cat = {k:v for k,v in remaining_cat_counts.items() if v > 0}
	prob_cat = 1. / len(nonzero_cat)
	probs = np.array([prob_cat / nonzero_cat[c[1]] for c in remaining_files])
	probs = probs / probs.sum() 
	inds = np.random.choice(range(len(remaining_files)), n_cat_training, replace=False, p=probs)
	for i in inds:
		flist['train'].append(remaining_files[i])

	print([(k, len(v)) for k, v in flist.items()])

	return flist


def transform_image(img):
	if args.mode == 'color_gray':
		mode = random.choice(['c','g'])

		if mode == 'g':
			tf = transforms.Compose([
				transforms.Resize((224,224)),
				transforms.Grayscale(num_output_channels=3),
				transforms.ToTensor(),
				transforms.ToPILImage(),
			])
		
		elif mode == 'c':
			tf = transforms.Compose([
				transforms.Resize((224,224)),
				transforms.ToTensor(),
				transforms.ToPILImage(),
			])

	elif args.mode == 'noise':
		mode = random.choice([0,0.04,0.16])
		tf = transforms.Compose([
			transforms.Resize((224,224)),
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			AddGaussianNoise(0., mode),
			transforms.ToPILImage(),
		])

	elif args.mode == 'blur':
		mode = random.choice([0,1.,3.])
		tf = transforms.Compose([
			transforms.Resize((224,224)),
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			AddGaussianBlur(7, mode),
			transforms.ToPILImage(),
		])

	img = tf(img)

	return img, str(mode)

if __name__ == '__main__':
	# create splits flist
	flist = create_splits_flist()

	for mode in ['train', '0', '1', '2', '3', '4']:
		if not os.path.exists(os.path.join(args.output_dir, mode)):
			os.makedirs(os.path.join(args.output_dir, mode), exist_ok=True)

		for j, (f, c) in enumerate(flist[mode]):
			img = Image.open(os.path.join(args.dataset_dir, f))
			
			img, md = transform_image(img)
			# img.save(os.path.join(args.output_dir, mode, md + '_' + c + '_' + f))
			img.save(os.path.join(args.output_dir, mode, str(j)+'_' + args.mode + '_' + md + '_' + c + '.JPEG'))