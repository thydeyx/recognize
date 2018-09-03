# -*- coding: utf-8 -*-

import sys
import os
import time
import shutil
import json

import numpy as np

from Preprocess import Image_Preprocess
from Preprocess import Normalization
from Model import Region_Classifier


def checkFile(fileName):
    if os.path.isfile(fileName):
        return True;
    else:
        print(fileName + "is not found!");
        exit();


def checkDir(fileName, creat=False):
    if os.path.isdir(fileName):
        if creat:
            shutil.rmtree(fileName);
            os.mkdir(fileName);
    else:
        if creat:
            os.mkdir(fileName);
        else:
            print(fileName + "is not found!");
            exit();


def main():
    if len(sys.argv) != 8:
        print('Usage: python3 ' + sys.argv[
            0] + ' train_image_dir_path train_gt_dir_path validation_image_dir_path validation_gt_dir_path vgg_model_path config_path output_dir_path\n');
        sys.exit(2);

    start = time.time();

    train_image_dir_path = sys.argv[1];
    train_gt_dir_path = sys.argv[2];
    validation_image_dir_path = sys.argv[3];
    validation_gt_dir_path = sys.argv[4];
    vgg_model_path = sys.argv[5];
    config_path = sys.argv[6];
    output_dir_path = sys.argv[7];

    os.environ["CUDA_VISIBLE_DEVICES"] = "3";

    shutil.rmtree(output_dir_path, ignore_errors=True);
    os.mkdir(output_dir_path);

    checkDir(train_image_dir_path, False);
    checkDir(train_gt_dir_path, False);
    checkFile(vgg_model_path);
    checkFile(config_path);

    with open(config_path, "r") as config_fp:
        config_dict = json.load(config_fp);

    image_height = config_dict["image_height"];
    image_width = config_dict["image_width"];
    region_height = config_dict["region_height"];
    region_width = config_dict["region_width"];
    region_batch_size = config_dict["region_batch_size"];
    basic_lr = config_dict["l_r"];
    step = config_dict["step"];

    train_imdb_obj = Image_Preprocess.Image_Database(train_image_dir_path, train_gt_dir_path, image_height, image_width);
    val_imdb_obj = Image_Preprocess.Image_Database(validation_image_dir_path, validation_gt_dir_path, image_height, image_width);

    print("Image Size: ({}, {}, {})".format(train_imdb_obj.image_height, train_imdb_obj.image_width, 3));

    # mean, std = Normalization.get_mean_variance(train_imdb_obj);
    # normal_para_npy_path = os.path.join(output_dir_path, "normal_para.npy");
    # np.save(normal_para_npy_path, (mean, std));
    #
    # train_imdb_obj = Normalization.normalize(train_imdb_obj, mean, std);
    # val_imdb_obj = Normalization.normalize(val_imdb_obj, mean, std);

    train_imdb_obj.prepare_region_data(region_height, region_width);
    val_imdb_obj.prepare_region_data(region_height, region_width);

    Region_Classifier.model_train(train_imdb_obj, val_imdb_obj, vgg_model_path, region_batch_size, basic_lr, step, output_dir_path);

    end = time.time();
    print("Total Time: {}s".format(end - start));


if __name__ == "__main__":
    main();