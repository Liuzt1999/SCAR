# Deep Metric Learning Assisted by Intra-variance in A Semi-supervised View of Learning

Official PyTorch implementation of [Deep Metric Learning Assisted by Intra-variance in A Semi-supervised View of Learning](https://arxiv.org/abs/2304.10941)

Our method is improved by [Self-Supervised Synthesis Ranking for Deep Metric Learning](https://ieeexplore.ieee.org/abstract/document/9598814)

An powerful loss function is used to constrain the ranking relationship of similar samples to mine the intra-class variance.

This repository provides source code of experiments on four datasets (CUB-200-2011, Cars-196, Stanford Online Products and In-shop) and pretrained models.

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Datasets

Download four public benchmarks for deep metric learning
 - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
 - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
 - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))
 - In-shop Clothes Retrieval ([Link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html))

## Results

Note that a sufficiently large batch size and good parameters resulted in better overall performance than that described in the paper.
Larger hash code size means better results with greater consumption.

### CUB-200-2011

| Method |   Backbone   | R@1  | R@2  | R@4  | R@8  |
| :----: | :----------: | :--: | :--: | :--: | :--: |
|  Ours  | Inception-BN | 68.8 | 79.4 | 86.9 | 91.7 |



### Cars-196

| Method |   Backbone   | R@1  | R@2  | R@4  | R@8  |
| :----: | :----------: | :--: | :--: | :--: | :--: |
|  Ours  | Inception-BN | 86.6 | 92.0 | 95.1 | 97.4 |


### Stanford Online Products

| Method |   Backbone   | R@1  | R@10  | R@100  | R@1000  |
| :----: | :----------: | :--: | :--: | :--: | :--: |
|  Ours  | Inception-BN | 79.4 | 91.0 | 96.4 | 98.8 |


### In-shop Clothes Retrieval

| Method |   Backbone   | R@1  | R@10  | R@20  | R@40  |
| :----: | :----------: | :--: | :--: | :--: | :--: |
|  Ours  | Inception-BN | 91.7 | 98.3 | 98.9 | 99.1 |

## Acknowledgements

Our code is modified and adapted on these great repositories:

- [PyTorch Metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [2003.13911\] Proxy Anchor Loss for Deep Metric Learning (arxiv.org)](https://arxiv.org/abs/2003.13911)


