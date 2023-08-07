# Contrast, Stylize and Adapt: Unsupervised Contrastive Learning Framework for Domain Adaptive Semantic Segmentation

## Overview

To overcome the domain gap between synthetic and real-world datasets, unsupervised domain adaptation methods have been proposed for semantic segmentation. Majority of the previous approaches have attempted to reduce the gap either at the pixel or feature level, disregarding the fact that the two components interact positively. To address this, we present **CON**trastive **FE**a**T**ure and p**I**xel alignment (CONFETI) for bridging the domain gap at both the pixel and feature levels using a unique contrastive formulation. We introduce well-estimated prototypes by including category-wise cross-domain information to link the two alignments: the pixel-level alignment is achieved using the jointly trained style transfer module with the **prototypical semantic consistency**, while the feature-level alignment is enforced to cross-domain features with the **pixel-to-prototype contrast**. Our extensive experiments demonstrate that our method outperforms existing state-of-the-art methods using DeepLabV2.

## Setup environment

```
conda create -n confeti python=3.8
conda activate confeti
pip install -r requirements.txt
```

## Testing

Download checkpoints and config files from [GoogleDrive](https://drive.google.com/drive/folders/1CaClev_jycGgwrlgdVrqh_qSIODEPsEE?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1_l8x-Yd80wFrLqVD9_Vd9A) (password: 74gd).
 
```python
python -m tools.test <cfg pth> <ckpt pth>
```

## Train

### Dataset preparation

Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/), [Cityscapes](https://www.cityscapes-dataset.com/) datasets and [SYNTHIA](https://synthia-dataset.net/downloads/) dataset.

Extract datasets to `data` folder. The folder structure should look like this:

```
data
├── cityscapes
│   ├── gtFine
│   ├── leftImg8bit
├── gta5
│   ├── images
│   └── labels
└── synthia
    ├── GT
    └── RGB
```

### Style transfer model

Put the style transfer model checkpoints in `checkpoints` folder. The folder structure should look like this:

```
checkpoints
└── gta2cs_1080_512_2nd
    ├── latest_net_G.pth
    |── latest_net_D.pth
    └── latest_net_F.pth
```

The model that we provide is the one that has already gone through the first round of training.


### Training

* First stage
```python
python run_experiments.py --config configs/confeti/confeti_1st.py --name gta2cs_1080_512 --continue_train
```

* Second stage
```python
python run_experiments.py --config configs/confeti/confeti_2nd.py --name gta2cs_1080_512_2nd --continue_train
```
