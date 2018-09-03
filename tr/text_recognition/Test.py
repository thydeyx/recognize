# -*- coding: utf-8 -*-

import sys
import os
import time
import shutil
import json

import numpy as np

from Preprocess import Image_Preprocess
from Preprocess import Normalization
from Prediction import Text_Localization
from Prediction import Model_Convert


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
    if len(sys.argv) != 6:
        print('Usage: python3 ' + sys.argv[
            0] + ' test_image_dir_path test_gt_dir_path model_dir_path normal_para_npy_path config_path\n');
        sys.exit(2);

    start = time.time();

    test_image_dir_path = sys.argv[1];
    test_gt_dir_path = sys.argv[2];
    model_dir_path = sys.argv[3];
    normal_para_npy_path = sys.argv[4];
    config_path = sys.argv[5];

    os.environ["CUDA_VISIBLE_DEVICES"] = "2";

    checkDir(test_image_dir_path, False);
    checkDir(test_gt_dir_path, False);
    checkDir(model_dir_path, False);
    # checkFile(normal_para_npy_path);
    checkFile(config_path);

    with open(config_path, "r") as config_fp:
        config_dict = json.load(config_fp);

    image_height = config_dict["image_height"];
    image_width = config_dict["image_width"];
    prob_thres = config_dict["prob_thres"];

    test_imdb_obj = Image_Preprocess.Image_Database(test_image_dir_path, test_gt_dir_path, image_height, image_width);
    print("Image Size: ({}, {}, {})".format(test_imdb_obj.image_height, test_imdb_obj.image_width, 3));

    mean, std = np.load(normal_para_npy_path);
    test_imdb_obj = Normalization.normalize(test_imdb_obj, mean, std);

    # Provide these to run freeze_graph:
    # Graph definition file, stored as protobuf TEXT
    graph_def_path = os.path.join(model_dir_path, "model.pbtxt");
    # Trained model's checkpoint name
    checkpoint_file = os.path.join(model_dir_path, "checkpoints/model.ckpt");
    # Frozen model's output name
    frozen_model_path = os.path.join(model_dir_path, "frozen_model.pb");

    # Output CoreML model path
    coreml_model_path = os.path.join(model_dir_path, "model.mlmodel");

    # Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
    output_node_name_list = ["predictions/prob", "predictions/bbox_pred"];
    remove_output_op_name = "conv_relu_output";
    remove_input_op_name = "conv_final";

    Model_Convert.freeze_model(graph_def_path, checkpoint_file, frozen_model_path, output_node_name_list, remove_output_op_name, remove_input_op_name);
    tf_prob, tf_bbox = Text_Localization.tf_metrics_report(test_imdb_obj, frozen_model_path, test_image_dir_path, prob_thres);

    # input_tensor_shapes = {"images:0": [1, 720, 960, 3]};  # batch size is 1
    # output_tensor_names = ["predictions/prob:0", "predictions/bbox_pred:0"];

    # coreml_model = Model_Convert.tf2coreml(frozen_model_path, coreml_model_path, input_tensor_shapes, output_tensor_names);
    # coreml_prob, coreml_bbox = Text_Localization.coreml_metrics_report(test_imdb_obj, coreml_model_path, test_image_dir_path, prob_thres);

    # print(tf_prob[0], coreml_prob[0], tf_prob[-1], coreml_prob[-1])
    # Text_Localization.max_relative_error(tf_prob, coreml_prob);
    # Text_Localization.max_relative_error(tf_bbox, coreml_bbox);

    end = time.time();
    print("Total Time: {}s".format(end - start));


if __name__ == "__main__":
    main();