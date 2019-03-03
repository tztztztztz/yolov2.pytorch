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
- [x] data augmentation
- [x] pretrained network
- [x] reorg layer
- [x] multi-scale training
- [ ] reproduce original paper's mAP

## Main Results

| | training set | test set | mAP@416 | mAP@544 |
| :--: | :--: | :--: | :--: | :--: |
|this repo|VOC2007+2012|VOC2007|72.7|74.6|
|original paper|VOC2007+2012|VOC2007|76.8|78.6|

Running time: ~19ms (52FPS) on GTX 1080


## Prerequisites
- python 3.5.x
- pytorch 0.4.1
- tensorboardX
- opencv3
- pillow

## Preparation

First clone the code

    git clone https://github.com/tztztztztz/yolov2.pytorch.git
    
Install dependencies

	pip install -r requirements.txt

Then create some folder

    mkdir output 
    mkdir data

## Demo

Download the pretrained weights

```
wget http://pjreddie.com/media/files/yolo-voc.weights
```

You can run the demo with `cpu` mode

    python demo.py

Or with `gpu` mode

    python demo.py --cuda true

## Training on PASCAL VOC

### Prepare the data

1. Download the training data.

    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    
    # download 2012 data
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    ```    


2. Extract the training data, all the data will be in one directory named `VOCdevit`. We use `$VOCdevit` to represent
the data root path

    ```bash
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    
    # 2012 data
    tar xvf VOCtrainval_11-May-2012.tar
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
    
    # mkdir VOCdevkit2012
    # cd VOCdevkit2012
    # ln -s $VOCdevit/VOC2012 VOC2012
    ```
    
### Download pretrained network

    cd yolov2.pytorch
    cd data
    mkdir pretrained
    cd pretrained
    wget https://pjreddie.com/media/files/darknet19_448.weights
    


### Train the model

    python train.py --cuda true
     
 If you want use multiple GPUs to accelerate the training. you can use the command below.
 
    python train.py --cuda true --mGPUs true

**NOTE**: Multi-scale training uses more GPU memory. If you have only one GPU with 8G memory, it's better to set `multi-scale=False` in `config/config.py`. See [link](https://github.com/tztztztztz/yolov2.pytorch/blob/master/config/config.py#L31).
    
    
## Testing 
 
    python test.py --cuda true

 
 

















