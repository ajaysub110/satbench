import os
import argparse
import pandas as pd

from ImagenetCOCOMapping import mappings

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/Users/ajay/Downloads/ILSVRC/Data/CLS-LOC/val')
parser.add_argument('--label_csv', type=str, default='/Users/ajay/Downloads/ILSVRC/ImageSets/CLS-LOC/LOC_val_solution.csv')
parser.add_argument('--output_dir', type=str, default='/Users/ajay/code/anytime-prediction-data/16ClassImagenet')
args = parser.parse_args()

def classify():
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir, exist_ok=True)

	df = pd.read_csv(args.label_csv)

	for i, row in df.iterrows():
		filename = row['ImageId'] + '.JPEG'
		wnetid = row['PredictionString'].split()[0]

		# find mappings key corresponding to wnetid value
		category = [k for k, v in mappings.items() if wnetid in v]

		if len(category) == 1:
			# make directory for category
			category_dir = os.path.join(args.output_dir, category[0])
			if not os.path.exists(category_dir):
				os.makedirs(category_dir, exist_ok=True)
			
			# copy file to category directory
			src = os.path.join(args.data_dir, filename)
			if os.path.exists(src):
				dst = os.path.join(category_dir, filename)
				os.system('cp {} {}'.format(src, dst))
			else:
				print('{} does not exist'.format(src))
		else:
			print("no or too many category with {} wnetid".format(wnetid))

		
if __name__ == '__main__':
	classify()