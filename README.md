# Probability Propagation for Point Cloud Segmentation (PP-PCS) 
Official Implementation of the **"Probability Propagation for Faster and Efficient Point Cloud Segmentation using a Neural Network (Accepted at PRL 2023)"**.

![Main Image](figure/main.jpg)

[[Paper]](https://doi.org/10.1016/j.patrec.2023.04.010)


# Updates & TODO Lists
- [] Update env install file 


# Getting Started
## Environment Setup
- Tested on RTX 3090 with python 3.7, pytorch 1.8.0, torchvision 0.9.0, CUDA 11.2

## Training PointNet++ from-scratch

1. Download original PointNet ++ repository(Pytorch Implementation)

```
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
```

2. Download the ShapeNet dataset from [[Here]](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) 

3. Trainig PointNet++ with ShapeNet
 - $ckp_root : where to save checkpoint 

```
cd Pointnet_Pointnet2_pytorch/
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir $ckp_root
```

## Checkpoints 
We provide a total of 15 checkpoints that represent the best performance achieved during 300 epochs. These checkpoints were generated using three different point cloud sampling methods (FPS, Poisson, Random) and five different numbers of sampled points (2048, 1024, 512, 256, 128).

```
cd checkpoints/
```


## How to Run Probability Propagation (PP)

1. Download code zip file and unzip or clone our repository

 - download the pretrained models in folder (PP-PCS/pretrained_models)
 - or you can train PointNet++ network from-scratch following upper discription

2. Evaluate part-segmentation with PP method

- $d_cutoff_start : start point of effective distance ratio for part-segmentation
- $d_cutoff_end : end point of effective distance ratio for part-segmentation 

```
cd PP-PCS

# example: evaluate the performance in the part-segmentation with the range of distance ratio 38 to 44 
python part_segmentation_w_pp.py --d_cutoff_start 38 --d_cutoff_end 44
```

3. Output
Epoch | Sampling method | number of points | probability mapping, distance ratio | Neural Network performance | Probability Propagation performance |
