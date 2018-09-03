#! /bin/bash

prefix=/home/sdb/maxi/Data/Image_Process/ICDAR2013
suffix=rpn

tensorboard --logdir="${prefix}/model_${suffix}/log" --port=8008
