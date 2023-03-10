# Getting Started

This page provides basic tutorials about the usage of ReDet.
For installation instructions, please see [INSTALL.md](INSTALL.md).


## Prepare DOTA dataset.
It is recommended to symlink the dataset root to `ReDet/data`.

Here, we give an example for single scale data preparation of DOTA-v1.5.

First, make sure your initial data are in the following structure.
```
data/dota15
├── train
│   ├──images
│   └──labelTxt
├── val
│   ├──images
│   └──labelTxt
└── test
    └──images
```
Split the original images and create COCO format json. 
```
python DOTA_devkit/prepare_dota1_5.py --srcpath path_to_dota --dstpath path_to_split_1024
```
Then you will get data in the following structure
```
dota15_1024
├── test1024
│   ├──DOTA_test1024.json
│   └──images
└── trainval1024
     ├──DOTA_trainval1024.json
     └──images
```
For data preparation with data augmentation, refer to "DOTA_devkit/prepare_dota1_5_v2.py"


## Prepare HRSC2016 dataset.

First, make sure your initial data are in the following structure.

```
data/HRSC2016
├── Train
│   ├──AllImages
│   └──Annotations
└── Test
│   ├──AllImages
│   └──Annotations
```

Then you need to convert HRSC2016 to DOTA's format, i.e., 
rename `AllImages` to `images`, convert xml `Annotations` to DOTA's `txt` format.
Here we provide a script from s2anet: [HRSC2DOTA.py](https://github.com/csuhan/s2anet/blob/original_version/DOTA_devkit/HRSC2DOTA.py). Now, your `data/HRSC2016` should contain the following folders.

```
data/HRSC2016
├── Train
│   ├──images
│   └── labelTxt
└── Test
    └── images
```

Then we need to generate `json` labels with COCO's format.
 
```
python DOTA_devkit/HRSC20162COCO.py
```


## Inference with pretrained models


### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.

Examples:

Assume that you have already downloaded the checkpoints to `work_dirs/`.

1. Test ReDet with 1 GPU.
```shell
python tools/test.py configs/ReDet/ReDet_re50_refpn_1x_dota15.py \
    work_dirs/ReDet_re50_refpn_1x_dota15/ReDet_re50_refpn_1x_dota15-7f2d6dda.pth \ 
    --out work_dirs/ReDet_re50_refpn_1x_dota15/results.pkl
```

2. Test ReDet with 4 GPUs.
```shell
./tools/dist_test.sh configs/ReDet/ReDet_re50_refpn_1x_dota15.py \
    work_dirs/ReDet_re50_refpn_1x_dota15/ReDet_re50_refpn_1x_dota15-7f2d6dda.pth \
    4 --out work_dirs/ReDet_re50_refpn_1x_dota15/results.pkl 
```

3. Parse results for [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html)
```
python tools/parse_results.py --config configs/ReDet/ReDet_re50_refpn_1x_dota15.py --type OBB
```

4. Test and evaluate ReDet on HRSC2016.
```shell
# generate results
python tools/test.py configs/ReDet/ReDet_re50_refpn_3x_hrsc2016.py \
    work_dirs/ReDet_re50_refpn_3x_hrsc2016/ReDet_re50_refpn_3x_hrsc2016-d1b4bd29.pth \ 
    --out work_dirs/ReDet_re50_refpn_3x_hrsc2016/results.pkl

# evaluation
# remeber to modify the results path in hrsc2016_evaluation.py
python DOTA_devkit/hrsc2016_evaluation.py
```

### Convert ReResNet+ReFPN to standard Pytorch layers

We provide a [script](tools/convert_ReDet_to_torch.py) to convert the pre-trained weights of ReResNet+ReFPN to standard Pytorch layers. Take ReDet on DOTA-v1.5 as an example.

1. download pretrained weights at [here](https://drive.google.com/file/d/1AjG3-Db_hmZF1YSKRVnq8j_yuxzualRo/view?usp=sharing), and convert it to standard pytorch layers.
```
python tools/convert_ReDet_to_torch.py configs/ReDet/ReDet_re50_refpn_1x_dota15.py \
        work_dirs/ReDet_re50_refpn_1x_dota15/ReDet_re50_refpn_1x_dota15-7f2d6dda.pth \
        work_dirs/ReDet_re50_refpn_1x_dota15/ReDet_r50_fpn_1x_dota15.pth
```

2. use standard ResNet+FPN as the backbone of ReDet and test it on DOTA-v1.5.
```
mkdir work_dirs/ReDet_r50_fpn_1x_dota15

bash ./tools/dist_test.sh configs/ReDet/ReDet_r50_fpn_1x_dota15.py \
        work_dirs/ReDet_re50_refpn_1x_dota15/ReDet_r50_fpn_1x_dota15.pth 8 \
        --out work_dirs/ReDet_r50_fpn_1x_dota15/results.pkl

# submit parsed results to the evaluation server.
python tools/parse_results.py --config configs/ReDet/ReDet_r50_fpn_1x_dota15.py
```

### Demo of inference in a large size image.


```python
python demo_large_image.py
```


## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

**\*Important\***: The default learning rate in config files is for 8 GPUs.
If you use less or more than 8 GPUs, you need to set the learning rate proportional
to the GPU num, e.g., 0.01 for 4 GPUs and 0.04 for 16 GPUs.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.


### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (recommended): Perform evaluation at every k (default=1) epochs during the training.
- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

### Train with multiple machines

If you run mmdetection on a cluster managed with [slurm](https://slurm.schedmd.com/), you can just use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [${GPUS}]
```

Here is an example of using 16 GPUs to train Mask R-CNN on the dev partition.

```shell
./tools/slurm_train.sh dev mask_r50_1x configs/mask_rcnn_r50_fpn_1x.py /nfs/xxxx/mask_rcnn_r50_fpn_1x 16
```

You can check [slurm_train.sh](tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
pytorch [launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility).
Usually it is slow if you do not have high speed networking like infiniband.
