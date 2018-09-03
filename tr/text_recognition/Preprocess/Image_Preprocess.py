# -*- coding: utf-8 -*-

import sys
import os
import re
import random

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.utils.extmath import cartesian
import cv2 as cv


class Image_Database(object):
    def __init__(self, image_dir_path, gt_dir_path, image_height, image_width):
        self.image_height = image_height;  # image height
        self.image_width = image_width;  # image width

        self.convmap_height = int(np.ceil(self.image_height / 16.));  # CNN feature map height
        self.convmap_width = int(np.ceil(self.image_width / 16.));  # CNN feature map width

        self.scale = np.array([1 / 16, 1 / 10, 1 / 8, 1 / 6, 1 / 4]);
        self.aspect_ratio = np.array([1, 2, 5, 8]);
        self.anchor_size = self.scale.shape[0] * self.aspect_ratio.shape[0];

        self.image_dict = dict();
        self.load_data(image_dir_path, gt_dir_path, self.image_dict);
        self.image_num = len(self.image_dict);
        self.fg_thres = 0.5;
        self.bg_thres = 0.2;
        self.bbox_normalize_scale = 5;

        self.proposal_prepare(self.image_dict);
        self.name_list = list(self.image_dict.keys());

        self.iter_index = 0;

    def get_name_dict(self, image_name_list, gt_name_list, image_dict):  # Get the name set from image and ground truth to ensure the same name.
        image_name_set = set();
        for image_name in image_name_list:
            image_name_set.add(os.path.splitext(image_name)[0]);

        gt_name_set = set();
        for gt_name in gt_name_list:
            gt_name_set.add(os.path.splitext(gt_name)[0][3:]);

        name_set = set.intersection(image_name_set, gt_name_set);

        for name in name_set:
            image_dict[name] = {"image": None, "gt": None, "proposal": None, "fgsize": None, "src_size": None};

    def load_image(self, image_dir_path, image_name_list, image_dict):  # Load images from directory to imdb dict.
        for image_name in image_name_list:
            image_path = os.path.join(image_dir_path, image_name);
            name = os.path.splitext(image_name)[0];

            if name in image_dict and os.path.isfile(image_path):
                im = cv.imread(image_path);
                image_dict[name]["src_size"] = (im.shape[1], im.shape[0]);
                im = cv.resize(im, (self.image_width, self.image_height));
                print(image_name, im.shape);
                pix = np.array(im).astype(np.float32);
                image_dict[name]["image"] = pix;

    def load_ground_truth(self, gt_dir_path, gt_name_list, image_dict):  # Load ground-truth from directory to imdb_dict
        for gt_name in gt_name_list:
            gt_path = os.path.join(gt_dir_path, gt_name);
            name = os.path.splitext(gt_name)[0][3:];

            if name in image_dict and os.path.isfile(gt_path) and re.match(".*.txt", gt_name):
                fp = open(gt_path, "r", encoding="UTF-8");
                gt_list = [];

                for line in fp:
                    line = line.strip();
                    if len(line) == 0:
                        continue;

                    line = line.replace(",", "");
                    line_split = re.split(" +|\t+", line);
                    pos_gt = [int(line_split[0]) * self.image_width / image_dict[name]["src_size"][0],
                              int(line_split[1]) * self.image_height / image_dict[name]["src_size"][1],
                              int(line_split[2]) * self.image_width / image_dict[name]["src_size"][0],
                              int(line_split[3]) * self.image_height / image_dict[name]["src_size"][1]];
                    # word_gt = line_split[4];

                    # gt_list.append([pos_gt, word_gt]);
                    gt_list.append(pos_gt);

                image_dict[name]["gt"] = np.array(gt_list, dtype=int);

    def load_data(self, image_dir_path, gt_dir_path, image_dict):  # Load Image Database
        image_name_list = os.listdir(image_dir_path);
        gt_name_list = os.listdir(gt_dir_path);

        # print(image_name_list);
        # print(gt_name_list);

        self.get_name_dict(image_name_list, gt_name_list, image_dict);
        self.load_image(image_dir_path, image_name_list, image_dict);
        self.load_ground_truth(gt_dir_path, gt_name_list, image_dict);

    def generate_anchors(self):  # Generate anchors from the specificed scale and ration.
        anchors = np.zeros([self.anchor_size, 4]);
        scale_size = self.scale.shape[0];
        aspect_ratio_size = self.aspect_ratio.shape[0];

        for i in range(scale_size):
            for j in range(aspect_ratio_size):
                anchor_height = int(self.image_height * self.scale[i]);
                anchor_width = int(anchor_height * self.aspect_ratio[j]);
                anchors[i * aspect_ratio_size + j, :] = np.array([-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height]);

        return anchors;

    def compute_overlap(self, mat1, mat2):  # Calculate the overlap area between proposals and ground truth
        s1 = mat1.shape[0];
        s2 = mat2.shape[0];
        area1 = (mat1[:, 2] - mat1[:, 0]) * (mat1[:, 3] - mat1[:, 1]);
        if mat2.shape[1] == 5:
            area2 = mat2[:, 4];
        else:
            area2 = (mat2[:, 2] - mat2[:, 0]) * (mat2[:, 3] - mat2[:, 1]);

        x1 = cartesian([mat1[:, 0], mat2[:, 0]]);

        x1 = np.amax(x1, axis=1);
        x2 = cartesian([mat1[:, 2], mat2[:, 2]]);
        x2 = np.amin(x2, axis=1);
        com_zero = np.zeros(x2.shape[0]);
        w = x2 - x1;
        w = w - 1;

        w = np.maximum(com_zero, w);

        y1 = cartesian([mat1[:, 1], mat2[:, 1]]);
        y1 = np.amax(y1, axis=1);
        y2 = cartesian([mat1[:, 3], mat2[:, 3]]);
        y2 = np.amin(y2, axis=1);
        h = y2 - y1;
        h = h - 1;
        h = np.maximum(com_zero, h);

        oo = w * h;

        aa = cartesian([area1[:], area2[:]]);
        aa = np.sum(aa, axis=1);

        ooo = oo / (aa - oo);

        overlap = np.transpose(ooo.reshape(s1, s2), (1, 0));

        return overlap;

    def compute_regression(self, roi, proposal):  # Calculate the regression target
        target = np.zeros(4);
        w1 = roi[2] - roi[0];
        h1 = roi[3] - roi[1];
        w2 = proposal[2] - proposal[0];
        h2 = proposal[3] - proposal[1];

        target[0] = (roi[0] - proposal[0]) / w2;
        target[1] = (roi[1] - proposal[1]) / h2;
        target[2] = np.log(w1 / w2);
        target[3] = np.log(h1 / h2);

        return target;

    def compute_target(self, roi_s, proposals_s):  # Get useful proposals for model training
        roi = roi_s.copy();
        proposals = proposals_s.copy();

        proposal_size = proposals.shape[0];
        roi_proposal_mat = np.zeros([proposal_size, 9]);

        if roi.shape[0] == 0:
            return roi_proposal_mat, 0;

        overlap = self.compute_overlap(roi, proposals);
        overlap_max = np.max(overlap, axis=1);
        overlap_max_idx = np.argmax(overlap, axis=1);
        fg_proposal_num = 0;

        for i in range(proposal_size):
            roi_proposal_mat[i, 1:5] = proposals[i, :];

            if self.proposals_mask[i] == 1:
                if overlap_max[i] >= self.fg_thres:
                    roi_proposal_mat[i, 0] = 1;
                    roi_proposal_mat[i, 5:] = self.compute_regression(roi[overlap_max_idx[i], :4], proposals[i, :]);
                    fg_proposal_num += 1;

                elif overlap_max[i] < self.bg_thres:
                    roi_proposal_mat[i, 0] = -1;

        return roi_proposal_mat, fg_proposal_num;

    def proposal_prepare(self, image_dict):  # Generate proposal positions for the original image
        anchors = self.generate_anchors();
        proposals = np.zeros([self.anchor_size * self.convmap_height * self.convmap_width, 4], dtype=int);

        for i in range(self.convmap_height):
            y0 = i * 16 + 8;
            for j in range(self.convmap_width):
                x0 = j * 16 + 8;
                for k in range(self.anchor_size):
                    index = (i * self.convmap_width + j) * self.anchor_size + k;
                    anchor = anchors[k, :];
                    proposals[index, :] = anchor + np.array([x0, y0, x0, y0]);

        self.proposals = proposals;
        # ignore cross-boundary anchors
        proposals_keep = np.where((proposals[:, 0] > 0) & (proposals[:, 1] > 0) & (proposals[:, 2] < self.image_width) & (proposals[:, 3] < self.image_height))[0];
        self.proposals_mask = np.zeros(proposals.shape[0]);
        self.proposals_mask[proposals_keep] = 1;
        self.proposal_size = self.proposals.shape[0];

        # area = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1]);
        # proposals = np.hstack([proposals, area.reshape([area.shape[0], 1])]);
        print("proposal size: {}".format(self.proposal_size));

        for image_name in image_dict:
            image_dict[image_name]["proposal"], image_dict[image_name]["fgsize"] = self.compute_target(image_dict[image_name]["gt"], proposals);
            print("image_name: {}, fgsize: {}".format(image_name, image_dict[image_name]["fgsize"]));

        # print("Compute Target: {}/{}".format(image_num, image_num));
        # self.fg_proposals_per_image = fg_proposal_num_dict;

    def generate_minibatch_proposal(self, image_batch_size, proposal_batch_size):  # Generate mini-batch for RPN model training
        pix_vec = np.zeros([image_batch_size, self.image_height, self.image_width, 3]);
        labels_vec = np.zeros([image_batch_size, self.proposal_size, 2]);
        labels_weight_vec = np.zeros([image_batch_size, self.proposal_size]);
        bbox_targets_vec = np.zeros([image_batch_size, self.proposal_size, 4]);
        bbox_loss_weight_vec = np.zeros([image_batch_size, self.proposal_size]);

        for i in range(image_batch_size):
            if self.iter_index == self.image_num:
                random.shuffle(self.name_list);
                self.iter_index = 0;

            while self.image_dict[self.name_list[self.iter_index]]["fgsize"] < 10:
                if self.iter_index == self.image_num - 1:
                    random.shuffle(self.name_list);
                    self.iter_index = 0;
                else:
                    self.iter_index += 1;

            cur_name = self.name_list[self.iter_index];
            im_train = self.image_dict[cur_name];

            pix = im_train["image"];

            roi_proposal = im_train["proposal"];

            fg_idx = np.where(roi_proposal[:, 0] == 1)[0];
            bg_idx = np.where(roi_proposal[:, 0] == -1)[0];

            labels = np.hstack([np.zeros([self.proposal_size, 1]), np.ones([self.proposal_size, 1])]);
            labels[fg_idx, 0] = 1;
            labels[fg_idx, 1] = 0;
            bbox_targets = roi_proposal[:, 5:];

            fg_num = int(min(fg_idx.shape[0], proposal_batch_size / 2));
            np.random.shuffle(fg_idx);
            fg_idx = fg_idx[:fg_num];
            bg_num = int(min(bg_idx.shape[0], proposal_batch_size - fg_num));
            np.random.shuffle(bg_idx);
            bg_idx = bg_idx[:bg_num];

            # fg_num = fg_idx.shape[0];
            # bg_num = bg_idx.shape[0];

            labels_weight = np.zeros(self.proposal_size);
            bbox_loss_weight = np.zeros(self.proposal_size);
            labels_weight[fg_idx] = bg_num / fg_num;
            labels_weight[bg_idx] = 1;
            bbox_loss_weight[fg_idx] = 1;

            pix_vec[i] = pix;
            labels_vec[i] = labels;
            labels_weight_vec[i] = labels_weight;
            bbox_targets_vec[i] = bbox_targets;
            bbox_loss_weight_vec[i] = bbox_loss_weight;

            self.iter_index += 1;

            # print(np.sum(labels_weight), np.sum(bbox_loss_weight));
            # print(labels);

        return pix_vec, labels_vec, labels_weight_vec, bbox_targets_vec, bbox_loss_weight_vec;

    def prepare_region_data(self, region_height, region_width): # Prepare data for region classifier
        self.pos_image_list = [];
        self.neg_image_list = [];
        self.region_height = region_height;
        self.region_width = region_width;

        for image_name in self.image_dict:
            pix = self.image_dict[image_name]["image"];
            gts = self.image_dict[image_name]["gt"];
            pos_num = gts.shape[0];

            for gt in gts:
                pos_img = pix[gt[1]:gt[3] + 1, gt[0]:gt[2] + 1].astype(np.uint8).copy();
                pos_img = cv.resize(pos_img, (region_width, region_height));
                self.pos_image_list.append(pos_img.astype(np.float32));

            roi_proposal = self.image_dict[image_name]["proposal"];
            bg_idx = np.where((roi_proposal[:, 0] != 1) & (roi_proposal[:, 1] > 0) & (roi_proposal[:, 2] > 0) & (roi_proposal[:, 3] < self.image_width) & (roi_proposal[:, 4] < self.image_height))[0];
            # bg_idx = np.where((roi_proposal[:, 0] == 0) & (self.proposals_mask == 1))[0];
            np.random.shuffle(bg_idx);
            bg_idx = bg_idx[:pos_num];

            for idx in bg_idx:
                neg_box = roi_proposal[idx, 1:5].astype(np.int);
                neg_img = pix[neg_box[1]:neg_box[3], neg_box[0]:neg_box[2]].astype(np.uint8).copy();
                neg_img = cv.resize(neg_img, (region_width, region_height));
                self.neg_image_list.append(neg_img.astype(np.float32));

        self.region_num = len(self.pos_image_list);
        self.region_index = 0;
        print("Region Num: {}".format(len(self.pos_image_list)));

    def generate_minibatch_region(self, region_batch_size): # Generate minibatch for region classifier training
        region_remain_num = int(region_batch_size / 2);
        pos_image_batch = [];
        neg_image_batch = [];

        while region_remain_num > 0:
            end_pos = min(self.region_index + region_remain_num, self.region_num);

            pos_image_batch.extend(self.pos_image_list[self.region_index:end_pos]);
            neg_image_batch.extend(self.neg_image_list[self.region_index:end_pos]);

            region_remain_num -= (end_pos - self.region_index);
            self.region_index = end_pos;

            if self.region_index == self.region_num:
                random.shuffle(self.pos_image_list);
                random.shuffle(self.neg_image_list);
                self.region_index = 0;

        pos_image_batch = np.array(pos_image_batch);
        neg_image_batch = np.array(neg_image_batch);

        image_batch = np.vstack([pos_image_batch, neg_image_batch]);

        pos_label_batch = np.hstack([np.ones([int(region_batch_size / 2), 1]), np.zeros([int(region_batch_size / 2), 1])]);
        neg_label_batch = np.hstack([np.zeros([int(region_batch_size / 2), 1]), np.ones([int(region_batch_size / 2), 1])]);
        label_batch = np.vstack([pos_label_batch, neg_label_batch]);

        return image_batch, label_batch;


def recover_pos(proposal, target):
    real_pos = np.zeros(4, dtype=int);
    w2 = proposal[2] - proposal[0];
    h2 = proposal[3] - proposal[1];

    real_pos[0] = target[0] * w2 + proposal[0];
    real_pos[1] = target[1] * h2 + proposal[1];

    w1 = np.exp(target[2]) * w2;
    h1 = np.exp(target[3]) * h2;

    real_pos[2] = real_pos[0] + w1;
    real_pos[3] = real_pos[1] + h1;

    return real_pos;


def main():
    image_dir_path = "/Users/max/Downloads/Data/Handwriting_Data/ICDAR2013/image_test/";
    gt_dir_path = "/Users/max/Downloads/Data/Handwriting_Data/ICDAR2013/gt_test/";
    dst_dir_path = "/Users/max/Downloads/Data/Presentation/weekly/"

    imdb_obj = Image_Database(image_dir_path, gt_dir_path, 720, 960);

    # print(imdb_obj.image_dict);
    # for image_name in imdb_obj.image_dict:
    #     im = cv.imread(os.path.join(image_dir_path, image_name) + ".jpg");
    #     im = cv.resize(im, (imdb_obj.image_width, imdb_obj.image_height));
    #     print(imdb_obj.image_dict[image_name]["image"].shape)
    #     #
    #     for gt in imdb_obj.image_dict[image_name]["gt"]:
    #         cv.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 3);
    #
    #     for proposal in imdb_obj.image_dict[image_name]["proposal"]:
    #         if proposal[0] == 1:
    #             cv.rectangle(im, (int(proposal[1]), int(proposal[2])), (int(proposal[3]), int(proposal[4])), (0, 255, 0), 3);
    #             real_pos = recover_pos(proposal[1:5], proposal[5:]);
    #             cv.rectangle(im, (real_pos[0], real_pos[1]), (real_pos[2], real_pos[3]),
    #                          (0, 0, 255), 3);
    #
    #     plt.imshow(im);
    #     plt.show();

        # cv.imwrite(os.path.join(dst_dir_path, image_name) + "_predict.png", im);
        # print(imdb_obj.image_dict[image_name]["fgsize"]);

    imdb_obj.prepare_region_data(224, 224);
    image_batch, label_batch = imdb_obj.generate_minibatch_region(32);
    print(len(image_batch));
    # for image in image_batch:
    #     image = image.astype(np.uint8);
    #     plt.imshow(image);
    #     plt.show();


if __name__ == "__main__":
    main();
