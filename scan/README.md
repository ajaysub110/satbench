# SCAN

We use code from the ![original implementation](https://github.com/ArchipLab-LinfengZhang/pytorch-scalable-neural-networks) and modified it for our experiments.

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## Training

Networks can be trained on 16-class ImageNet dataset using the following command:

```bash
python train.py --tag trainSCAN_MODE_COLOR --mode MODE --data imagenet --class_num 16 --epoch 200
```

where noise is 'color', 'gray', 'noise' or 'blur'.

## Testing

Trained networks can be tested on the same images used to test humans, using the following command:

```bash
python test.py --tag testSCAN_MODE_COLOR --mode MODE --data imagenet --class_num 16 --load_path LOAD_PATH
```

where MODE is 'color', 'gray', 'noise' or 'blur' and LOAD_PATH is the path to the saved model.
