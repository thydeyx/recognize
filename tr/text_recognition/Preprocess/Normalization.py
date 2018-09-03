import numpy as np


def get_mean_variance(train_imdb_obj):
    pix_sum = np.zeros(3);
    image_num = 0;

    for image_name in train_imdb_obj.image_dict:
        pix = train_imdb_obj.image_dict[image_name]["image"];
        pix_sum += np.sum(pix, axis=(0, 1, 2));
        image_num += 1;

    mean = pix_sum / (image_num * train_imdb_obj.image_height * train_imdb_obj.image_width);
    var_sum = np.zeros(3);

    for image_name in train_imdb_obj.image_dict:
        pix = train_imdb_obj.image_dict[image_name]["image"];
        var_sum += np.sum((pix - mean) ** 2, axis=(0, 1, 2));

    std = np.sqrt(var_sum / (image_num * train_imdb_obj.image_height * train_imdb_obj.image_width));

    return mean, std;


def normalize(imdb_obj, mean, std):
    for image_name in imdb_obj.image_dict:
        pix = imdb_obj.image_dict[image_name]["image"];
        imdb_obj.image_dict[image_name]["image"] = (pix - mean) / std;

    return imdb_obj;
