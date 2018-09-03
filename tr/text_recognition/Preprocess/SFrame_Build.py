import re
import os

import turicreate as tc


def bbox_coordinates_convert(pos_gt):
    """
    Takes a row and returns a dictionary representing bounding
    box coordinates:  (center_x, center_y, width, height)  e.g. {'x': 100, 'y': 120, 'width': 80, 'height': 120}
    """

    center_x = (pos_gt[0] + pos_gt[2]) / 2;
    center_y = (pos_gt[1] + pos_gt[3]) / 2;

    width = pos_gt[2] - pos_gt[0];
    height = pos_gt[3] - pos_gt[1];

    return {"x": center_x, "width": width, "y": center_y, 'height': height};


def load_image(image_dir_path):  # Load images from directory to imdb dict.
    # Load all images in random order
    image_sf = tc.image_analysis.load_images(image_dir_path, recursive=True, random_order=True);

    # Split path to get filename
    image_sf['name'] = image_sf['path'].apply(lambda path: os.path.basename(path).split(".")[0]);

    # Original path no longer needed
    del image_sf['path'];

    return image_sf;


def load_ground_truth(gt_dir_path, gt_name_list):  # Load ground-truth from directory to SFrame
    gt_dict = {"name": [], "annotations": []};

    for gt_name in gt_name_list:
        gt_path = os.path.join(gt_dir_path, gt_name);
        name = os.path.splitext(gt_name)[0][3:];

        if os.path.isfile(gt_path) and re.match(".*.txt", gt_name):
            fp = open(gt_path, "r", encoding="UTF-8");
            gt_list = [];

            for line in fp:
                line = line.strip();
                if len(line) == 0:
                    continue;

                line = line.replace(",", "");
                line_split = re.split(" +|\t+", line);
                pos_gt = [int(line_split[0]), int(line_split[1]), int(line_split[2]), int(line_split[3])];
                coordinates = bbox_coordinates_convert(pos_gt);
                label = "text";

                # word_gt = line_split[4];

                gt_list.append({"label": label, "coordinates": coordinates});

            gt_dict["name"].append(name);
            gt_dict["annotations"].append(gt_list);

    gt_sf = tc.SFrame(gt_dict);
    return gt_sf;


def load_data(image_dir_path, gt_dir_path):  # Load Image Database
    gt_name_list = os.listdir(gt_dir_path);

    image_sf = load_image(image_dir_path);
    gt_sf = load_ground_truth(gt_dir_path, gt_name_list);

    # Join annotations with the images. Note, some images do not have annotations,
    # but we still want to keep them in the dataset. This is why it is important to
    # a LEFT join.
    data_sf = image_sf.join(gt_sf, on="name", how="left");

    # The LEFT join fills missing matches with None, so we replace these with empty
    # lists instead using fillna.
    data_sf["annotations"] = data_sf["annotations"].fillna([]);

    return data_sf;
