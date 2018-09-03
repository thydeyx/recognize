import os
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import cv2 as cv
import coremltools

from Prediction import NMS


def recover_pos(proposals, targets):
    real_pos = np.zeros([proposals.shape[0], 4]);
    w2 = proposals[:, 2] - proposals[:, 0];
    h2 = proposals[:, 3] - proposals[:, 1];

    real_pos[:, 0] = targets[:, 0] * w2 + proposals[:, 0];
    real_pos[:, 1] = targets[:, 1] * h2 + proposals[:, 1];

    w1 = np.exp(targets[:, 2]) * w2;
    h1 = np.exp(targets[:, 3]) * h2;

    real_pos[:, 2] = real_pos[:, 0] + w1;
    real_pos[:, 3] = real_pos[:, 1] + h1;

    return real_pos;


def rpn_nms(prob, target_preds, proposals, x_rng, y_rng):
    pos_pred = recover_pos(proposals, target_preds);
    bbox = np.zeros([proposals.shape[0], 5]);

    bbox[:, :4] = pos_pred;
    bbox[:, 4] = prob;

    bbox = NMS.filter_bbox(bbox, x_rng, y_rng);
    # bbox = NMS.non_max_suppression_slow(bbox, 0.5);
    bbox = NMS.non_max_suppression_fast(bbox, 0.5);

    if len(bbox) == 0:
        return bbox;

    keep_prob = np.sort(bbox[:, 4])[max(-50, -1 * bbox.shape[0])];

    index = np.where(bbox[:, 4] >= keep_prob)[0];
    bbox = bbox[index];

    return bbox;


def cal_IoU(rect1, rect2):
    if rect1[0] > rect2[2] or rect2[0] > rect1[2] or rect1[1] > rect2[3] or rect2[1] > rect1[3]:
        return 0;

    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]);
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]);

    area_intersection = (min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) * (min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    area_union = area1 + area2 - area_intersection;

    return area_intersection / area_union;


def cal_metrics(bbox, gts, overlap_thres, prob_thres):
    bbox_num = bbox.shape[0];
    gts_num = gts.shape[0];

    right_vec = np.zeros(gts_num, dtype=int);
    overlap_vec = np.zeros(gts_num);
    predict_num = 0;

    for i in range(bbox_num):
        if bbox[i][4] > prob_thres:
            predict_num += 1;

            for j in range(gts_num):
                overlap = cal_IoU(bbox[i, :4], gts[j]);

                if overlap > overlap_thres and overlap > overlap_vec[j]:
                    right_vec[j] = 1;
                    overlap_vec[j] = overlap;

    tp_num = np.sum(right_vec);

    return tp_num, predict_num, gts_num;


def load_graph(frozen_graph_path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_path, "rb") as fp:
        graph_def = tf.GraphDef();
        graph_def.ParseFromString(fp.read());

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph;


def tf_metrics_report(test_imdb_obj, frozen_model_path, test_image_dir_path, prob_thres):
    tf.reset_default_graph();
    x_rng = [0.1, test_imdb_obj.image_width - 0.1];
    y_rng = [0.1, test_imdb_obj.image_height - 0.1];

    print("Testing Begin");

    overlap_thres = test_imdb_obj.fg_thres;

    tp_num = 0;
    predict_num = 0;
    gts_num = 0;

    graph = load_graph(frozen_model_path);

    prob_list = [];
    bbox_list = [];

    with tf.Session(graph=graph) as sess:
        # tf.saved_model.loader.load(sess, ["RPN"], model_dir_path);

        cls_prob = sess.graph.get_tensor_by_name("predictions/prob:0");
        bbox_pred = sess.graph.get_tensor_by_name("predictions/bbox_pred:0");

        for image_name in test_imdb_obj.image_dict:
            pix = test_imdb_obj.image_dict[image_name]["image"];
            proposals = test_imdb_obj.image_dict[image_name]["proposal"][:, 1:5];
            gts = test_imdb_obj.image_dict[image_name]["gt"];

            pix = np.expand_dims(pix, axis=0)

            (prob, target_preds) = sess.run([cls_prob, bbox_pred], feed_dict={"images:0": pix});
            prob = np.squeeze(prob);
            target_preds = np.squeeze(target_preds);
            prob_list.append(prob);
            bbox_list.append(target_preds);

            tf_bbox = rpn_nms(prob[:, 0], target_preds, proposals, x_rng, y_rng);

            if len(tf_bbox) != 0:
                tp_num_add, predict_num_add, gts_num_add = cal_metrics(tf_bbox, gts, overlap_thres, prob_thres);
                tp_num += tp_num_add;
                predict_num += predict_num_add;
                gts_num += gts_num_add;

            # im = cv.imread(os.path.join(test_image_dir_path, image_name) + ".jpg");
            # im = cv.resize(im, (test_imdb_obj.image_width, test_imdb_obj.image_height));
            #
            # for gt in gts:
            #     cv.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 3);
            #
            # for box in bbox:
            #     cv.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3);

            # plt.imshow(im);
            # plt.show();
            # cv.imwrite(os.path.join(dst_dir_path, image_name) + "_predict.png", im);

    precision = tp_num / predict_num;
    recall = tp_num / gts_num;

    if precision == 0 and recall == 0:
        f1_score = 0;
    else:
        f1_score = 2 * (precision * recall) / (precision + recall);

    print("Tensorflow ~ Precision: {}, Recall: {}, F1-Score: {}".format(precision, recall, f1_score));

    return np.array(prob_list), np.array(bbox_list);


def coreml_metrics_report(test_imdb_obj, coreml_model_path, test_image_dir_path, prob_thres):
    x_rng = [0.1, test_imdb_obj.image_width - 0.1];
    y_rng = [0.1, test_imdb_obj.image_height - 0.1];

    print("Testing Begin");

    overlap_thres = test_imdb_obj.fg_thres;

    tp_num = 0;
    predict_num = 0;
    gts_num = 0;

    coreml_model = coremltools.models.MLModel(coreml_model_path);

    prob_list = [];
    bbox_list = [];

    for image_name in test_imdb_obj.image_dict:
        pix = test_imdb_obj.image_dict[image_name]["image"];
        proposals = test_imdb_obj.image_dict[image_name]["proposal"][:, 1:5];
        gts = test_imdb_obj.image_dict[image_name]["gt"];

        pix = np.transpose(pix, axes=[2, 0, 1])[None, None, :, :, :];
        inputs = {'images__0': pix};
        output = coreml_model.predict(inputs, useCPUOnly=False);

        prob = np.transpose(np.squeeze(output["predictions__prob__0"]));
        target_preds = np.transpose(np.squeeze(output["predictions__bbox_pred__0"]));

        prob_list.append(prob);
        bbox_list.append(target_preds);

        coreml_bbox = rpn_nms(prob[:, 0], target_preds, proposals, x_rng, y_rng);

        if len(coreml_bbox) != 0:
            tp_num_add, predict_num_add, gts_num_add = cal_metrics(coreml_bbox, gts, overlap_thres, prob_thres);
            tp_num += tp_num_add;
            predict_num += predict_num_add;
            gts_num += gts_num_add;

        # im = cv.imread(os.path.join(test_image_dir_path, image_name) + ".jpg");
        # im = cv.resize(im, (test_imdb_obj.image_width, test_imdb_obj.image_height));
        #
        # for gt in gts:
        #     cv.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 3);
        #
        # for box in bbox:
        #     cv.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3);

        # plt.imshow(im);
        # plt.show();
        # cv.imwrite(os.path.join(dst_dir_path, image_name) + "_predict.png", im);

    precision = tp_num / predict_num;
    recall = tp_num / gts_num;

    if precision == 0 and recall == 0:
        f1_score = 0;
    else:
        f1_score = 2 * (precision * recall) / (precision + recall);

    print("CoreML ~ Precision: {}, Recall: {}, F1-Score: {}".format(precision, recall, f1_score));

    return np.array(prob_list), np.array(bbox_list);


def max_relative_error(x,y):
    den = np.maximum(x,y)
    den = np.maximum(den,1)
    rel_err = (np.abs(x-y))/den
    max_rel_error = np.max(rel_err);
    print("Max relative error: {}".format(max_rel_error));

    return max_rel_error;



