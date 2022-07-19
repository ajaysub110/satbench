# Generating images for human & network evaluation

## 1. Download ILSVRC validation dataset
1. Download `imagenet_object_localization_patched2019.tar.gz` and `LOC_val_solution.csv` from the ILSVRC competition 2012 subset available [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
2. Unzip the downloaded file
3. We will only be using the files available in `CLS-LOC/val` directory and `LOC_val_solution.csv`.

## 2. Create the 16-class ImageNet dataset from the ILSVRC validation dataset
Run `generate_images/create16ClassImagenet.py` providing the path to the `CLS-LOC/val`, `LOC_val_solution.csv` file and output directory as arguments.
```
python generate_images/create16ClassImagenet.py \
	--data_dir=<PATH2valdirectory> \
	--label_csv <PATH2csv> \
	--output_dir=<PATH2outputdirectory>
```
This will move images from the val dataset to subfolders organised by higher-level categories.

## 3. Select images for human/network evaluation for different time conditions given perturbation type
Run `generate_images/create_humanSplits.py` to create human/network test set.
```
cd generate_images
python create_humanSplits.py \
	--data_dir=<PATH216ClassImageNet> \
	--output_dir =<PATH2outputdirectory> \
	--n_modes=<number of perturbation values> \
	--n_rts=<number of reaction time blocks> \
	--mode=<perturbation type. Options: color_gray, noise, blur> \
```
The above command will generate the test set for the given perturbation type, organized into subfolders by reaction time. For our experiments, `n_rts` was 5. The folders will be named 0-4. 0 was used for the lowest RT and 4 for highest. Additionally, a train folder with 50 images used for training human observers on categorization will be generated.