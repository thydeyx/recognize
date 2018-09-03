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


def load_data(image_dir_path, gt_dir_path):  # Load Image Database
    gt_name_list = os.listdir(gt_dir_path);

    image_sf = SFrame_Build.load_image(image_dir_path);
    gt_sf = SFrame_Build.load_ground_truth(gt_dir_path, gt_name_list);

    # Join annotations with the images. Note, some images do not have annotations,
    # but we still want to keep them in the dataset. This is why it is important to
    # a LEFT join.
    data_sf = image_sf.join(gt_sf, on="name", how="left");

    # The LEFT join fills missing matches with None, so we replace these with empty
    # lists instead using fillna.
    data_sf["annotations"] = data_sf["annotations"].fillna([]);

    return data_sf;


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 ' + sys.argv[0] + ' test_image_dir_path test_gt_dir_path model_dir_path\n");
        sys.exit(2);

    start = time.time();

    test_image_dir_path = sys.argv[1];
    test_gt_dir_path = sys.argv[2];
    model_dir_path = sys.argv[3];

    checkDir(test_image_dir_path, False);
    checkDir(test_gt_dir_path, False);

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2";

    # Use all GPUs (default)
    tc.config.set_num_gpus(-1);

    test_data_sf = load_data(test_image_dir_path, test_gt_dir_path);

    model_path = os.path.join(model_dir_path, "TextDetector.model");
    model = tc.load_model(model_path);

    # Save predictions to an SArray
    predictions = model.predict(test_data_sf);

    # Evaluate the model and save the results into a dictionary
    metrics = model.evaluate(test_data_sf);

    print(metrics);

    # # Export for use in Core ML
    # model.export_coreml("TextDetector.mlmodel");

    end = time.time();
    print("Total Time: {}s".format(end - start));


if __name__ == "__main__":
    main();