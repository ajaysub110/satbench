# SATBench: Benchmarking the speed-accuracy tradeoff in object recognition by humans and dynamic neural networks

**This repository contains code used in the paper:**

![SATBench: Benchmarking the speed-accuracy tradeoff in object recognition by humans and dynamic neural networks](https://arxiv.org/abs/2206.08427) by Ajay Subramanian, Omkar Kumbhar, Elena Sizikova, Najib J. Majaj, Denis G. Pelli (New York University, 2022).

Our contributions are as follows:
* We present large-scale (148 human observers), public dataset on timed ImageNet [1] object recognition with 16 categories, across color, grayscale, 3 noise and 3 blur conditions. For each condition, we tested human performance for 5 reaction time (RT) values. This data provides a benchmark for the human speed-accuracy tradeoff and is specifically intended to facilitate comparison between neural networks and humans on timed object recognition.
* We present comparable benchmarks for dynamic neural networks, a class of networks capable of inference-time adaptive computation.
* We perform an extensive quantitative comparison between speed-accuracy tradeoffs in humans and four dynamic neural networks. To do so, we propose three novel metrics: RMSE between SAT curves, category-wise correlation, and steepness which ease model-human comparison.

![](assets/human-network-sat.jpeg)

## Table of Contents
1. Dataset
2. Code
3. Citation
4. References

## Dataset
Our human dataset is collected using a reaction time paradigm proposed by McElree & Carrasco [2] where observers are forced to respond at a beep which sounds at a specific time after target presentation. Varying the beep interval across several blocks helps us collect object recognition data across different reaction times (`500ms`, `900ms`, `1100ms`, `1300ms`, `1500ms`). We evaluate dynamic neural networks using the same paradigm with computational FLOPs used as an analog for reaction time.

Human dataset and network results can be found at https://osf.io/2cpmb/. Download and unzip `human-data.zip` and `model_data.zip` for human and network data respectively.

## Code
1. Code to generate image dataset used in the paper is available in the `generate_images` directory.
2. JSON files corresponding to LabJS studies used to collect human data are available in the `human_data_collection` directory.
3. Code used to analyze human data is available as notebooks in the `human_data_analysis` directory.
4. We benchmark 4 dynamic neural network models - MSDNet [2] , SCAN [3], Cascaded-Nets (CNets) [4]  and ConvRNN [3] on our dataset. The following table mentions the scripts to be used for training and inference of each model. We used code for each model from existing/official implementations (links given below):
  - MSDNet: https://github.com/kalviny/MSDNet-PyTorch
  - SCAN: https://github.com/ArchipLab-LinfengZhang/pytorch-scalable-neural-networks
  - ConvRNN: https://github.com/cjspoerer/rcnn-sat
  - CNet: https://github.com/michael-iuzzolino/CascadedNets

  Code for each model is available in a subdirectory with the model's name. Links to pretrained networks will be added to respective README.md files upon publication.

## Citation
```
@article{subramanian2022satbench,
  title={SATBench: Benchmarking the speed-accuracy tradeoff in object recognition by humans and dynamic neural networks},
  author={Subramanian, Ajay and Price, Sara and Kumbhar, Omkar and Sizikova, Elena and Majaj, Najib J and Pelli, Denis G},
  journal={arXiv preprint arXiv:2206.08427},
  year={2022}
}
```

## Reference
[1] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei, L. (2015). Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3), 211-252.

[2] Huang, G., Chen, D., Li, T., Wu, F., Van Der Maaten, L., & Weinberger, K. Q. (2017). Multi-scale dense networks for resource efficient image classification. arXiv preprint arXiv:1703.09844.

[3] Zhang, L., Tan, Z., Song, J., Chen, J., Bao, C., & Ma, K. (2019). Scan: A scalable neural networks framework towards compact and efficient models. Advances in Neural Information Processing Systems, 32.

[4] Iuzzolino, M., Mozer, M. C., & Bengio, S. (2021). Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss. Advances in Neural Information Processing Systems, 34.

[5] Spoerer, C. J., Kietzmann, T. C., Mehrer, J., Charest, I., & Kriegeskorte, N. (2020). Recurrent neural networks can explain flexible trading of speed and accuracy in biological vision. PLoS computational biology, 16(10), e1008215.
