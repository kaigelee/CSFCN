# CSFCN
By Kaige Li, Qichuan Geng, Maoxian Wan, Xiaochun Cao, Senior Member, IEEE, and Zhong Zhou. This repository is an official implementation of the paper "Context and Spatial Feature Calibration for Real-Time Semantic Segmentation", which is under review. The full code will be released after review.

## Highlights
<p align="center">
  <img src="figs/city_score.png" alt="overview-of-our-method" width="500"/></br>
  <span align="center">Comparison of inference speed and accuracy for real-time models on test set of Cityscapes.</span> 
</p>

* **Towards Real-time Applications**: S2FCN could be directly used for the real-time applications, such as autonomous vehicle and medical imaging.
* **A Novel and Efficient Decoder**: a novel and efficient, Context and Spatial Decoder is introduced to deal with the problems of pixel-context mismatch and spatial
feature misalignment via pooling-based and sampling-based attention mechanisms.
* **More Accurate and Faster**: CSFCN presents 78.7% mIOU with speed of 70.0 FPS on Cityscapes test set and 77.8% mIOU with speed of 179.2 FPS on CamVid test set.

## 🎉 Updates 🎉

Good News! In the follow-up experiments, we found that higher performance can be achieved by **removing Gaussian blur** during data augmentation. (The experiment is still in progress, the final mIoU may be higher than 78.7%)! Now, the data augmentation process includes:

- [ ] random cropping
- [ ] random horizontal flipping 
- [ ] random scaling
- [ ] color jitter



## Demos

A demo of the segmentation performance of our proposed CSFCNs: Predictions of CSFCN-100 (left), CSFCN-75 (middle), and CSFCN-50 (right).
<p align="center">
  <img src="figs/video0_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #0</span>
</p>

<p align="center">
  <img src="figs/video1_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #1</span>
</p>

<p align="center">
  <img src="figs/video2_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #2</span>
</p>

## Overview
<p align="center">
  <img src="figs/csfcn.png" alt="overview-of-our-method" width="800"/></br>
  <span align="center">An overview of the basic architecture of our proposed Context and Spatial Feature Calibration Network (CSFCN). </span> 
</p>
CFC and SFC blocks are responsiable for context feature calibration, spatial feature calibration, respectively.

## Metrics

:bell: We append 50, 75 and 100 after the network name to represent the input sizes of 512 × 1024, 768 × 1536 and 1024 × 2048, respectively.

| Model (Cityscapes) | Val (% mIOU) |Test (% mIOU)| FPS (GTX 1080 Ti) |
|:-:|:-:|:-:|:-:|
| CSFCN-50 | 74.0 | 73.8 | 229.1 |
| CSFCN-75 | 77.3 |77.2 | 122.2  |
| CSFCN-100 | 79.0 |78.7 | 70.0 |

:bell: Our Cityscapes pre-trained CSFCN obtains 81.0% mIoU on the CamVid set.

| Model (CamVid) | Val (% mIOU) | Test (% mIOU)| FPS (GTX 1080 Ti) |
|:-:|:-:|:-:|:-:|
| CSFCN |-| 77.8 | 179.2 |
| CSFCN-P |-| 81.0 | 179.2 |


## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/CSFCN
source ~/venv/CSFCN/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```


## Usage

### 0. Prepare the dataset

* Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets and unzip them in `data/cityscapes` and `data/camvid` dirs.


### 1. Training

* Download the ImageNet pretrained models and put them into `pretrained_models/imagenet/` dir.
* For example, train the CSFCN on Cityscapes with batch size of 12 on one GPU (e.g., 3090):
````bash
python tools/train.py --cfg configs/CSFCN_cityscapes.yaml
````
* Or train the CSFCN on Cityscapes using train and val sets simultaneously with batch size of 12 on one GPU:
````bash
python tools/trainval.py --cfg configs/CSFCN_cityscapes_trainval.yaml
````

### 2. Evaluation

* Download the finetuned models for Cityscapes and CamVid and put them into `pretrained_models/cityscapes/` and `pretrained_models/camvid/` dirs, respectively.
* For example, evaluate the CSFCN on Cityscapes val set:
````bash
python tools/eval.py --cfg configs/CSFCN_cityscapes.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/CSFCN_best_model.pt
````
* Or, evaluate the CSFCN on CamVid test set:
````bash
python tools/eval.py --cfg configs/CSFCN_camvid.yaml \
                          TEST.MODEL_FILE pretrained_models/camvid/CSFCN_camvid_best_model.pt
````
* Generate the testing results of CSFCN on Cityscapes test set:
````bash
python tools/submit.py --cfg configs/CSFCN_cityscapes_trainval.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/CSFCN_trainval_best_model.pt 
````

### 3. Speed Measurement

#### 3.0 Latency measurement tools

* If you have successfully installed [TensorRT](https://github.com/chenwydj/FasterSeg#installation), you will automatically use TensorRT for the following latency tests (see [function](https://github.com/chenwydj/FasterSeg/blob/master/tools/utils/darts_utils.py#L167) here).
* Otherwise you will be switched to use Pytorch for the latency tests  (see [function](https://github.com/chenwydj/FasterSeg/blob/master/tools/utils/darts_utils.py#L184) here).

#### 3.1 Measure the speed of the CSFCN

* Measure the inference speed of CSFCN-100 for Cityscapes:
````bash
python models/speed/CSFCN_speed.py --c 19 --r 1024 2048
````
* Measure the inference speed of CSFCN for CamVid:
````bash
python models/speed/CSFCN_speed.py --c 11 --r 720 960
````

### 4. Custom Inputs

* Put all your images in `samples/` and then run the command below using Cityscapes pretrained CSFCN for image format of .png:
````bash
python tools/custom.py --p '../pretrained_models/cityscapes/CSFCN_best_model.pth' --t '*.png'
````


## TODO
- [ ] Refactor and clean code
- [ ] Organize all codes and upload them
- [ ] Release complete config, network and training files


## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [ICNet-pytorch](https://github.com/liminn/ICNet-pytorch)
* [PIDNet](https://github.com/XuJiacong/PIDNet)
* [DDRNet](https://github.com/ydhongHIT/DDRNet)

