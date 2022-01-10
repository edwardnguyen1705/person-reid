# Deep Person Re-identification

## Supported

- Backbone
  - Resnet
    - Resnet
    - SeResnet
    - Resnext
    - Resnet-D
    - ResNest
  - OSNet
  - Efficientnet b0-b4
  - MobilenetV3
  - OSNet-AIN
- Attention
  - CbamModule
  - FPBAttention (FPB: Feature Pyramid Branch for Person Re-Identification)
  - GlobalContext
  - NonLocalBlock
  - PLRAttention (Learning Diverse Features with Part-Level Resolution for Person Re-Identification)
  - SEModule
- GEM Pooling
- CosSoftmax, CircleSoftmax, ArcFaceSoftmax (FastReID: A Pytorch Toolbox for General Instance Re-identification)
- Loss function
  - Center Loss
  - Cricle Loss
  - Cosface Loss
  - Smooth Label - Cross Entropy loss
  - Focal Loss
  - MultiSimilarity Loss
  - Hard Mining Triplet Loss
  - Weighted Triplet Loss
- WarmupMultiStepLR, WarmupCosineAnnealingLR
- Transform
  - AutoAugment
  - Cutout
  - LGT (Local Grayscale Transfomation: An Effective Data Augmentation for person re-identification
    )
  - Random2DTranslation
  - RandomErasing
- Data Prefetcher, Prefetch Generator

## Installation

### Requirements

- python >= 3.8
- pytorch >= 1.9
- torchvision
- opencv
- wandb

### Setup with pip

- Create new python enviroment
- `pip3 install -r requirements.gpu.txt -f https://download.pytorch.org/whl/torch_stable.html`

## Getting started

### Download and extract datasets

#### Market-1501

- Download dataset from [kaggle.com/pengcw1/market-1501](https://www.kaggle.com/pengcw1/market-1501/data)
- Extract zip file:

```
    root/ # args.data_root
    |--bounding_box_test/
    |--bounding_box_train/
    |--gt_bbox/
    |--query/
```

### Setup cython to accelerate evalution

- `cd src/metrics/rank_cylib`
- `make all`

### Training and Evaluation

```
python src/engine/train.py --cfg [PATH_CONFIG] --gpu [DEVICE_ID] --checkpoint_dir [FOLDER_TO_SAVE_CHECKPOINT] --val --val-step 10 --test-from-checkpoint
```

## Results

- [LightMBN](https://github.com/jixunbo/LightMBN) + [GlobalContext](https://arxiv.org/pdf/2012.13375v1.pdf) + GeneralizedMeanPooling:

  - mAP: 91.7%
  - mINP: 74.11%

- Baseline + CircleSoftmax + BNNeck + GeneralizedMeanPoolingP + NonLocalBlock(ratio=0.0625) + AutoAugmentation(p=0.1)

  - mAP: 90.41%
  - mINP: 71.04%

Other pretrained result: [wandb](https://wandb.ai/hiennguyen9874/rep-reid-v2)

## TODO

- More datasets: DukeMTMC-reID, MSMT17, CUHK03, Market1501-500k
- Heatmap visualize
- Light Reid
- ReRanking & GPU ReRanking
- Re-train with fixed resnet version (docs/fixed_resnet.png)
