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

## News

:bell: Good News! I achieved an mIoU of [78.2998](https://www.cityscapes-dataset.com/anonymous-results/?id=4b20f9c0105bedd70b73d5e841ec523f6950314bb65db013c3311674c88e1428) in the newest experbiment by removing Gaussian blur during data augmentation. (the experiment is still running and the final mIoU may be even higher)! Now, the data augmentation process includes

- [ ] random cropping
- [ ] random horizontal flipping 
- [ ] random scaling
- [ ] color jitter



## Demos

A demo of the segmentation performance of our proposed CSFCNs: Predictions of CSFCN-100 (left) and CSFCN-50 (right).
<p align="center">
  <img src="figs/video1_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #1</span>
</p>

<p align="center">
  <img src="figs/video2_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #2</span>
</p>

## Metrics

Because we plan to embed our method into the robot designed by our research group to improve its ability to understand the scene. Therefore, we plan to migrate our CSFCN to TensorRT, and test the speed on embedded systems NVIDIA Jetson AGX Xavier and NVIDIA GTX 1080Ti, respectively.

| Model (Cityscapes) | Val (% mIOU) | Test (% mIOU)| FPS (RTX 3090) | FPS (Xavier&TensorRT) |  FPS (1080Ti&TensorRT) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| CSFCN-50 |  73.7 | 72.7 | 309.7 | pending | pending |
| CSFCN-75 | 77.6  | 76.6 | 167.3 | pending | pending |
| CSFCN-100 | 79.1 | 78.1 | 109.0 | pending | pending |

| Model (CamVid) | Val (% mIOU) | Test (% mIOU)| FPS (RTX 3090) |
|:-:|:-:|:-:|:-:|
| CSFCN |-| 77.2 | 250.4 |


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
* For example, train the CSFCN on Cityscapes with batch size of 12 on one GPU:
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
                          TEST.MODEL_FILE pretrained_models/cityscapes/CSFCN_Cityscapes_test.pt
````
* Or, evaluate the CSFCN on CamVid test set:
````bash
python tools/eval.py --cfg configs/CSFCN_camvid.yaml \
                          TEST.MODEL_FILE pretrained_models/camvid/CSFCN_Camvid_test.pt
````
* Generate the testing results of CSFCN on Cityscapes test set:
````bash
python tools/eval.py --cfg configs/CSFCN_cityscapes_trainval.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/CSFCN_trainval_Cityscapes_test.pt 
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
python tools/custom.py --p '../pretrained_models/cityscapes/CSFCN_Cityscapes_test.pth' --t '*.png'
````


## TODO
- [ ] Refactor and clean code
- [ ] Organize all codes and upload them
- [ ] Release complete config, network and training files


## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
* [ICNet-pytorch](https://github.com/liminn/ICNet-pytorch)
* [PIDNet](https://github.com/XuJiacong/PIDNet)
* [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch)
* [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
* [DDRNet](https://github.com/ydhongHIT/DDRNet)

