import re
import os
import sys
import time
import shutil
import json

import turicreate as tc

from Preprocess import SFrame_Build


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
        print(
            "Usage: python3 ' + sys.argv[0] + ' train_image_dir_path train_gt_dir_path validation_image_dir_path validation_gt_dir_path output_dir_path\n");
        sys.exit(2);

    start = time.time();

    train_image_dir_path = sys.argv[1];
    train_gt_dir_path = sys.argv[2];
    validation_image_dir_path = sys.argv[3];
    validation_gt_dir_path = sys.argv[4];
    output_dir_path = sys.argv[5];

    shutil.rmtree(output_dir_path, ignore_errors=True);
    os.mkdir(output_dir_path);

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2";

    checkDir(train_image_dir_path, False);
    checkDir(train_gt_dir_path, False);

    # Use all GPUs (default)
    tc.config.set_num_gpus(-1);

    train_data_sf = SFrame_Build.load_data(train_image_dir_path, train_gt_dir_path);
    validation_data_sf = SFrame_Build.load_data(validation_image_dir_path, validation_gt_dir_path);

    train_data_path = os.path.join(output_dir_path, "train_data.sframe");
    validation_data_path = os.path.join(output_dir_path, "validation_data.sframe");

    # Save SFrame
    train_data_sf.save(train_data_path);
    validation_data_sf.save(validation_data_path);

    # Load the data
    train_data_sf = tc.SFrame(train_data_path);
    validation_data_sf = tc.SFrame(validation_data_path);

    print("SFrame Completed");
    print(train_data_sf);

    # Create a model
    model = tc.object_detector.create(train_data_sf, feature="image", annotations="annotations", batch_size=128);
    print("Model Completed");

    # Save the model for later use in Turi Create
    model_dir_path = os.path.join(output_dir_path, "model");
    os.mkdir(model_dir_path);
    model_path = os.path.join(model_dir_path, "TextDetector.model");
    model.save(model_path);

    # Export for use in Core ML
    model.export_coreml(os.path.join(model_dir_path, "TextDetector.mlmodel"));

    end = time.time();
    print("Total Time: {}s".format(end - start));


if __name__ == "__main__":
    main();