#! /bin/bash

prefix=/home/sdb/maxi/Data/Image_Process/ICDAR2013
suffix=rpn

train_image_dir_path=$prefix/Training_Task1_Images/
train_gt_dir_path=$prefix/Training_Task1_GT/
validation_image_dir_path=$prefix/Validate_Task1_Images/
validation_gt_dir_path=$prefix/Validate_Task1_GT/
test_image_dir_path=$prefix/Test_Task1_Images/
test_gt_dir_path=$prefix/Test_Task1_GT/

#train_image_dir_path=$prefix/image_test/
#train_gt_dir_path=$prefix/gt_test/
#validation_image_dir_path=$prefix/image_test/
#validation_gt_dir_path=$prefix/gt_test/
#test_image_dir_path=$prefix/image_test/
#test_gt_dir_path=$prefix/gt_test/

vgg_model_path=$prefix/others/vgg16.npy
output_dir_path=$prefix/model_rpn_${suffix}
model_dir_path=$output_dir_path/model/
normal_para_npy_path=$output_dir_path/normal_para.npy

config_path=./rpn_config.json

python text_recognition/Test.py ${test_image_dir_path} ${test_gt_dir_path} ${model_dir_path} ${normal_para_npy_path} ${config_path} > log/test_log_${suffix}.txt

