import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import cv2 as cv
import coremltools


def load_graph(frozen_graph_path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_path, "rb") as fp:
        graph_def = tf.GraphDef();
        graph_def.ParseFromString(fp.read());

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph;


def tf_metrics_report(test_imdb_obj, frozen_model_path):
    tf.reset_default_graph();
    print("Testing Begin");

    graph = load_graph(frozen_model_path);
    image_batch, label_batch = test_imdb_obj.generate_minibatch_region(test_imdb_obj.region_num * 2);
    tp_num = 0;
    predict_num = 0;
    gts_num = test_imdb_obj.region_num;

    with tf.Session(graph=graph) as sess:
        # tf.saved_model.loader.load(sess, ["RPN"], model_dir_path);

        cls_prob = sess.graph.get_tensor_by_name("predictions/prob:0");

        for i in range(len(image_batch)):
            image = image_batch[i];
            image = np.expand_dims(image, axis=0)

            prob = sess.run([cls_prob], feed_dict={"images:0": image});
            prob = np.squeeze(prob);

            if np.argmax(prob) == 0:
                predict_num += 1;
                if np.argmax(label_batch[i]) == 0:
                    tp_num += 1;

    precision = tp_num / predict_num;
    recall = tp_num / gts_num;

    if precision == 0 and recall == 0:
        f1_score = 0;
    else:
        f1_score = 2 * (precision * recall) / (precision + recall);

    print("Tensorflow ~ Precision: {}, Recall: {}, F1-Score: {}".format(precision, recall, f1_score));



