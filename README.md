# HW2

## Abstract

In this work, I use EfficientDet to train my model

EfficientDet [Paper](https://arxiv.org/abs/1911.09070)|[Github](https://github.com/signatrix/efficientdetO)

## Reproducing Submission

To reproduct my submission without retrainig, do the following steps

1. [Installation](#installation)
2. [Download Official Image](#download-official-image)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation

```bash
python -m pip install -r requirements.txt
```

## Dataset Prepare

### Prepare Images

After downloading, the data directory is structured as

```
data
    +- train
    +- test
```

### Download Official Image

Download and extract train.zip and test.zip to dataraw directory.

#### Split Dataset

The dataset will split to train and val atuomatically.

## Training

My final submission is use efficientDet 

Run `train.py` to train.

```bash
python train.py
```

The expected training times are

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
efficientDet | 2x 2080Ti | 128 | 120 | 12 hours

## Pretrained models

You can download pretrained model that used for my submission from [link](https://drive.google.com/file/d/1fVJKKIEKPPtnXXvjnh-99dPp6LlTBMu-/view).

Unzip them into results then you can see following structure

```bash

+- model.pth
```

## Inference

If trained weights are prepared, you can create the result file which named result.json by run below command

```bash
$python test.py --resume={trained model path}
```

## Visualize the result

If you want to visualize the prediction result, you can use the module "draw_picture", and the result will in "./picture"

the function: def draw(processed_img, itr, bbox, label, score)
processed_img is the image dataloader load.
itr is the number of current image
bbox is the predicted bounding box
label is the predicted bounding label
score is the predicted bounding score

## Make Submission

Click [here](https://reurl.cc/Z7olyA) to submission the json file!!

## Citation

```

@article{EfficientDetSignatrix,
    Author = {Signatrix GmbH},
    Title = {A Pytorch Implementation of EfficientDet Object Detection},
    Journal = {https://github.com/signatrix/efficientdet},
    Year = {2020}
}
```