## Unsupervised Domain Adaptation for Remote Sensing Semantic Segmentation with Transformer

Our article has been published in Remote Sensing as part of
the Special Issue Deep Learning for Satellite Image Segmentation!

[HTML Version](https://www.mdpi.com/2072-4292/14/19/4942/htm) | [PDF Version](https://www.mdpi.com/2072-4292/14/19/4942/pdf)

If you find this project useful in your research, please consider citing:

```
Li, W.; Gao, H.; Su, Y.; Momanyi, B.M. Unsupervised Domain Adaptation for Remote Sensing Semantic Segmentation with Transformer. Remote Sens. 2022, 14, 4942. https://doi.org/10.3390/rs14194942
```

## Preparation

### Pre-requisites

- Python >= 3.8

- Pytorch >= 1.7.1

- mmcv-full >= 1.3.7

### Pretrained weights & Checkpoints

1. Please download the [MiT weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) pretrained on ImageNet-1K provided by the official [SegFormer repository](https://github.com/NVlabs/SegFormer) and put them in a folder `pretrained/` within this project. Only `mit_b5.pth` are uesd in our experiments.

2. We provide three checkpoints for each domain adaptation on POT→VAI and VAI→POT. Please put them in `work_dirs/`.

POT→VAI:

| ID              | Road  | Building | Vegetation | Tree  | Car   | Clutter | mIoU  | url                                                               |
| --------------- | ----- | -------- | ---------- | ----- | ----- | ------- | ----- | ----------------------------------------------------------------- |
| 20220804_231646 | 72.64 | 90.96    | 57.82      | 72.76 | 44.21 | 39.09   | 62.91 | [link](https://1drv.ms/u/s!Avyk1WkEG_5JvU2v6_ximQN61KJL?e=PRAUSj) |
| 20220804_215704 | 76.90 | 89.04    | 63.92      | 76.30 | 41.93 | 34.53   | 63.77 | [link](https://1drv.ms/u/s!Avyk1WkEG_5JvUxSEvCXlHvO0jDD?e=zqKVoH) |
| 20220809_094520 | 76.11 | 90.95    | 62.58      | 74.53 | 38.74 | 35.17   | 63.01 | [link](https://1drv.ms/u/s!Avyk1WkEG_5JvVFWeyKnab9RxYX2?e=AIl7AS) |
| **Mean**        | 75.22 | 90.32    | 61.44      | 74.53 | 41.63 | 36.26   | 63.23 |                                                                   |

VAI→POT:

| ID              | Road  | Building | Vegetation | Tree  | Car   | Clutter | mIoU  | url                                                               |
| --------------- | ----- | -------- | ---------- | ----- | ----- | ------- | ----- | ----------------------------------------------------------------- |
| 20220805_164511 | 61.16 | 73.33    | 56.34      | 61.72 | 66.12 | 1.16    | 53.3  | [link](https://1drv.ms/u/s!Avyk1WkEG_5JvU7wvXCSDH-JJJ0H?e=fuhxHq) |
| 20220805_180439 | 76.83 | 87.17    | 61.38      | 55.94 | 65.24 | 1.16    | 57.81 | [link](https://1drv.ms/u/s!Avyk1WkEG_5JvU8eD531wMbnWDco?e=MoWzXe) |
| 20220808_143609 | 75.47 | 87.54    | 60.86      | 52.24 | 65.23 | 0.11    | 56.91 | [link](https://1drv.ms/u/s!Avyk1WkEG_5JvVDVJsy5gPjbiNON?e=HKd6Vw) |
| **Mean**        | 71.15 | 82.68    | 59.53      | 56.63 | 65.53 | 0.81    | 56.01 |                                                                   |

### Setup Datasets

ISPRS Potsdam and ISPRS Vaihingen can be requested at [link](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx). 

For ISPRS **Potsdam** dataset, the ‘2_Ortho_RGB.zip’ and ‘5_Labels_all_noBoundary.zip’ are required. Please run the following command to re-organize the dataset.

```shell
python tools/convert_datasets/potsdam.py /path/to/potsdam
```

For ISPRS **Vaihingen** dataset, the ‘ISPRS_semantic_labeling_Vaihingen.zip’ and ‘ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip’ are required. Please run the following command to re-organize the dataset.

```shell
python tools/convert_datasets/vaihingen.py /path/to/vaihingen
```

After re-organizing, the datasets have the following structures:

```
.
├── ...
├── data
│   ├── potsdam
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   └── val
│   │   └── img_dir
│   │       ├── train
│   │       └── val
│   └── vaihingen
│       ├── ann_dir
│       │   ├── train
│       │   └── val
│       └── img_dir
│           ├── train
│           └── val
├── ...
```

For more information about datasets, please refer to [Prepare datasets](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html) provided by MMSegmentation.

### Training

To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance. A training job can be launched using:

```shell
python -m tools.train "configs/uda_rs/potsdam2isprs_uda_pt7_local7_label_warm_daformer_mitb5.py" # POT to VAI
python -m tools.train "configs/uda_rs/isprs2potsdam_uda_pt7_local7_label_warm_daformer_mitb5.py" # VAI to POT
```

By default, logs and checkpoints are stored in `work_dirs/<experiments>` with this structure:

```
work_dirs/<experiments>/<config_name>.py  # config file
work_dirs/<experiments>/latest.pth        # checkpoint 
work_dirs/<experiments>/<log_time>.log    # log
```

### Testing & Predictions

A testing on the validation set can be launched using:

```shell
python -m tools.test <CONFIG_FILE> <CHECKPOINT_FILE> --eval mIoU mFscore --show-dir <SHOW_DIR> --opacity 1 --gpu-id <GPU_ID>
```

For convenience, we provide a script to simplify the arguments:

```shell
test.sh work_dirs/<experiments> [iteration_num, [GPU_ID]] # By default, iteration 4000 and GPU 0 are used.
test.sh work_dirs/20220804_231646_potsdam2isprs_uda_pt7_dw_local7_label_warm_daformer_mitb5
```

The predictions are saved for inspection to `work_dirs/<experiments>/preds` and the mIoU of the model is printed to the console.

### Acknowledgements

This project is heavily based on the following open-source projects. We thank their authors for making the source code publically available.

- [DAFormer](https://github.com/lhoyer/DAFormer)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [DACS](https://github.com/vikolss/DACS)
