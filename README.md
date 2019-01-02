## Yolov2 Pytorch Implementation

This repository aims to learn and understand the YOLO algorithm. I am a beginner of deep learning, and I found the best way to learn a deep learning algorithm is to implement it from scratch. So if you also feel this way, just follow this repo! The code in this projects is clear and easier to understand, and I also documented it as much as possible. 

<div style="color:#0000FF" align="center">
<img src="images/result3.png" width="350"/>
<img src="images/result1.png" width="350"/> 
</div>

## Purpose

- [x] train pascal voc
- [x] multi-GPUs support
- [x] test
- [x] pascal voc validation
- [ ] data augmentation
- [ ] pretrained network
- [ ] reorg layer
- [ ] multi-scale training
- [ ] reproduce original paper's mAP

## Current Stage

I trained the network with PASCAL VOC 2007 train set data. No pretrained weights is used, Neither is data augmentation.

The reported results is as follows:

| train data | test data | mAP |
| ------------- | ------------- | ------------- |
| VOC2007 train | VOC2007 train   | 84.7|
| VOC2007 train  | VOC2007 test  | 10.7|

Obviously, this model is in the overfitting situation. In another word, it is a low bias and high variance model. I guess the reason is that the number of data is too small, and I don't apply data augmentation and transfer learning. However, it proves our model is correct at least, right?

I will keep on updating this repo in the next few months.

## Prerequisites
- python 3.5.x
- pytorch 0.4.1
- tensorboardX
- opencv3

## Preparation

First clone the code

    git clone https://github.com/tztztztztz/yolov2.pytorch.git

Then create some folder

    mkdir output 
    mkdir data

## Demo

Download the pretrained weights [Dropbox](https://www.dropbox.com/s/tqje3dctoaxbgmk/yolov2_epoch_160.pth?dl=0)

Place the weights file in the `$ROOT/output` folder

You can run the demo with `cpu` mode

    python demo.py

Or with `gpu` mode

    python demo.py --cuda true


### Prepare the data

1. Download the training data.

    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```    


2. Extract the training data, all the data will be in one directory named `VOCdevit`. We use `$VOCdevit` to represent
the data root path

    ```bash
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    ```

3. It should have this basic structure

    ```
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    ```

4. Create symlinks for the PASCAL VOC dataset

    ```
    cd yolov2.pytorch
    mkdir data
    cd data
    mkdir VOCdevkit2007
    cd VOCdevkit2007
    ln -s $VOCdevit/VOC2007 VOC2007
    ```
 
## Training
 
    python train.py --cuda true
     
 if you want use multiple GPUs to accelerate the training. you can use the command below.
 
    python train.py --cuda true --mGPUs true
    
    
## Testing 
 
    python test.py --cuda true
 
 

















