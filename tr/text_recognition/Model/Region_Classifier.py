import os
import sys
import time
import shutil

import numpy as np
import tensorflow as tf


class Region_Classifier:
    def __init__(self, vgg16_npy_path, image_height, image_width):
        self.data_dict = np.load(vgg16_npy_path, encoding="latin1").item();
        print("Vgg16 npy file loaded");

        self.VGG_MEAN = [103.939, 116.779, 123.68];

        self.image_height = image_height;
        self.image_width = image_width;

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, name="filter");
        return tf.Variable(initial);

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape);
        return tf.Variable(initial);

    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name="filter");

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases");

    def get_conv_filter_const(self, name):
        return tf.constant(self.data_dict[name][0], name="filter");

    def get_bias_const(self, name):
        return tf.constant(self.data_dict[name][1], name="biases");

    def avg_pool(self, x, name):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name);

    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name);

    def conv_layer(self, x, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name);
            conv = tf.nn.conv2d(x, filt, [1, 1, 1, 1], padding="SAME");

            bias = self.get_bias(name);
            conv_biases = tf.nn.bias_add(conv, bias);

            relu = tf.nn.relu(conv_biases);
            weight_dacay = tf.nn.l2_loss(filt, name="weight_dacay");
            return relu, weight_dacay;

    def conv_layer_const(self, x, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter_const(name);
            conv = tf.nn.conv2d(x, filt, [1, 1, 1, 1], padding="SAME");

            bias = self.get_bias_const(name);
            conv_biases = tf.nn.bias_add(conv, bias);

            relu = tf.nn.relu(conv_biases);
            return relu, 0;

    def conv_layer_new(self, x, name, kernel_size, out_channel=512, stddev=0.01):
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()[-1];
            # filt = tf.Variable(
            #     tf.random_normal([kernel_size[0], kernel_size[1], shape, out_channel], mean=0.0, stddev=stddev),
            #     name="filter")
            # conv_biases = tf.Variable(tf.zeros([out_channel]), name="biases")

            filt = self.weight_variable([kernel_size[0], kernel_size[1], shape, out_channel]);
            bias = self.bias_variable([out_channel]);

            conv = tf.nn.conv2d(x, filt, [1, 1, 1, 1], padding="SAME");
            conv_biases = tf.nn.bias_add(conv, bias);

            weight_dacay = tf.nn.l2_loss(filt, name="weight_dacay");
            return conv_biases, weight_dacay;

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var);
            tf.summary.scalar('mean', mean);
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
            tf.summary.scalar('stddev', stddev);
            tf.summary.scalar('max', tf.reduce_max(var));
            tf.summary.scalar('min', tf.reduce_min(var));
            tf.summary.histogram('histogram', var);

    def build(self, images, labels, learning_rate, keep_prob):
        start_time = time.time();
        print("build model started");

        # # Convert RGB to BGR
        # red, green, blue = tf.split(3, 3, rgb)
        # assert red.get_shape().as_list()[1:] == [image_height, image_width, 1]
        # assert green.get_shape().as_list()[1:] == [image_height, image_width, 1]
        # assert blue.get_shape().as_list()[1:] == [image_height, image_width, 1]
        # bgr = tf.concat([
        #     blue - self.VGG_MEAN[0],
        #     green - self.VGG_MEAN[1],
        #     red - self.VGG_MEAN[2],
        # ], 3)

        assert images.get_shape().as_list()[1:] == [self.image_height, self.image_width, 3];

        # VGG mean pixel subtracted
        images = images - self.VGG_MEAN;

        with tf.name_scope("vgg16"):
            # Conv layer 1
            self.conv1_1, conv1_1_wd = self.conv_layer_const(images, name="conv1_1");
            self.conv1_2, conv1_2_wd = self.conv_layer_const(self.conv1_1, name="conv1_2");
            self.weight_dacay = conv1_1_wd + conv1_2_wd;
            self.pool1 = self.max_pool(self.conv1_2, name="pool1");
            # Conv layer 2
            self.conv2_1, conv2_1_wd = self.conv_layer_const(self.pool1, name="conv2_1");
            self.conv2_2, conv2_2_wd = self.conv_layer_const(self.conv2_1, name="conv2_2");
            self.weight_dacay = conv2_1_wd + conv2_2_wd;
            self.pool2 = self.max_pool(self.conv2_2, name="pool2");
            # Conv layer 3
            self.conv3_1, conv3_1_wd = self.conv_layer_const(self.pool2, name="conv3_1");
            self.conv3_2, conv3_2_wd = self.conv_layer_const(self.conv3_1, name="conv3_2");
            self.conv3_3, conv3_3_wd = self.conv_layer_const(self.conv3_2, name="conv3_3");
            self.weight_dacay = conv3_1_wd + conv3_2_wd + conv3_3_wd;
            self.pool3 = self.max_pool(self.conv3_3, name="pool3");
            # Conv layer 4
            self.conv4_1, conv4_1_wd = self.conv_layer_const(self.pool3, name="conv4_1");
            self.conv4_2, conv4_2_wd = self.conv_layer_const(self.conv4_1, name="conv4_2");
            self.conv4_3, conv4_3_wd = self.conv_layer_const(self.conv4_2, name="conv4_3");
            self.weight_dacay += conv4_1_wd + conv4_2_wd + conv4_3_wd;
            self.pool4 = self.max_pool(self.conv4_3, name="pool4");
            # Conv layer 5
            self.conv5_1, conv5_1_wd = self.conv_layer_const(self.pool4, name="conv5_1");
            self.conv5_2, conv5_2_wd = self.conv_layer_const(self.conv5_1, name="conv5_2");
            self.conv5_3, conv5_3_wd = self.conv_layer_const(self.conv5_2, name="conv5_3");
            self.weight_dacay += conv5_1_wd + conv5_2_wd + conv5_3_wd;

        with tf.name_scope("batch_norm"):
            self.conv_relu = tf.nn.relu(self.conv5_3, name="conv_relu");
            self.conv_relu_norm = tf.layers.batch_normalization(self.conv_relu, name="conv_relu_norm");

        self.conv_relu_output = tf.identity(self.conv_relu_norm, name="conv_relu_output");

        with tf.name_scope("dropout"):
            self.conv_relu_dropout = tf.nn.dropout(self.conv_relu_output, keep_prob);

        self.conv_final = tf.identity(self.conv_relu_dropout, name="conv_final");

        with tf.name_scope("predictions"):
            self.conv_shape = self.conv_final.get_shape().as_list();
            flatten_size = self.conv_shape[1] * self.conv_shape[2] * self.conv_shape[3];
            self.fc_input = tf.reshape(self.conv_final, [-1, flatten_size], name="reshape");

            self.fc_weight = self.weight_variable([flatten_size, 2]);
            self.fc_bias = self.bias_variable([2]);

            self.cls_score = tf.matmul(self.fc_input, self.fc_weight) + self.fc_bias
            self.prob = tf.nn.softmax(self.cls_score, name="prob");
            self.weight_dacay += tf.nn.l2_loss(self.fc_weight, name="weight_dacay");

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                                          logits=self.cls_score));
            self.loss = self.cross_entropy + 0.005 * self.weight_dacay;
            self.cross_entropy_summary = tf.summary.scalar("cross_entropy", self.cross_entropy);
            self.weight_dacay_summary = tf.summary.scalar("weight_dacay", self.weight_dacay);
            self.loss_summary = tf.summary.scalar("loss", self.loss);

        with tf.name_scope("optimmizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
            with tf.control_dependencies(update_ops):
                # self.train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss);
                self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss);

        self.data_dict = None;
        print("build model finished: %ds" % (time.time() - start_time));

    def save(self, model_dir_path, sess, saver):
        # shutil.rmtree(model_dir_path, ignore_errors=True);
        # os.mkdir(model_dir_path);
        # builder = tf.saved_model.builder.SavedModelBuilder(model_dir_path);
        # builder.add_meta_graph_and_variables(sess, tags=["RPN"]);
        # builder.save();

        ckpt_dir_path = os.path.join(model_dir_path, "checkpoints");
        shutil.rmtree(ckpt_dir_path, ignore_errors=True);
        os.mkdir(ckpt_dir_path);
        ckpt_path = os.path.join(ckpt_dir_path, "model.ckpt");
        saver.save(sess, ckpt_path);


def model_train(train_imdb_obj, val_imdb_obj, vgg_model_path, region_batch_size, basic_lr, step, output_dir_path):
    tf.reset_default_graph();

    print_time = 1;
    save_time = 10;

    log_dir_path = os.path.join(output_dir_path, "log");
    model_dir_path = os.path.join(output_dir_path, "model");

    with tf.Session() as sess:
        # Inputs
        images = tf.placeholder(tf.float32, shape=[None, train_imdb_obj.region_height, train_imdb_obj.region_width, 3], name="images");
        learning_rate = tf.placeholder(tf.float32, name="learning_rate");
        keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob");

        # Outputs
        labels = tf.placeholder(tf.float32, [None, 2], name="labels");

        model = Region_Classifier(vgg_model_path, train_imdb_obj.region_height, train_imdb_obj.region_width);
        model.build(images, labels, learning_rate, keep_prob);

        train_summary_op = tf.summary.merge_all();
        train_writer = tf.summary.FileWriter(os.path.join(log_dir_path, "train"), sess.graph);

        val_summary_op = tf.summary.merge_all();
        val_writer = tf.summary.FileWriter(os.path.join(log_dir_path, "validation"), sess.graph);

        # Use a saver to save checkpoints
        saver = tf.train.Saver();

        sess.run(tf.global_variables_initializer());
        for var in tf.trainable_variables():
            print(var.name, var.get_shape().as_list(), sess.run(tf.nn.l2_loss(var)));

        print("Training Begin");
        start_time = time.time();

        train_loss_list = [];
        train_cross_entropy_list = [];

        val_loss_best = sys.maxsize;
        val_loss_list = [];
        val_cross_entropy_list = [];
        tf.train.write_graph(sess.graph, model_dir_path, "model.pbtxt");

        for i in range(step):
            batch = train_imdb_obj.generate_minibatch_region(region_batch_size);
            if i <= 7000:
                l_r = basic_lr;
            else:
                if i <= 9000:
                    l_r = basic_lr * 0.1;
                else:
                    l_r = basic_lr * 0.01;

            (train_summary, _, train_loss_iter, train_cross_entropy_iter, cls) = sess.run(
                [train_summary_op, model.train_step, model.loss, model.cross_entropy, model.cls_score],
                feed_dict={images: batch[0], labels: batch[1], learning_rate: l_r, keep_prob: 0.5});

            train_writer.add_summary(train_summary, global_step=i);
            train_loss_list.append(train_loss_iter);
            train_cross_entropy_list.append(train_cross_entropy_iter);

            if i % print_time == 0:
                print(
                    "step: {}, time: {}s, batch_shape: {}, loss: {}, cls_cross_entropy: {}, l_r: {}".format(
                        i, time.time() - start_time, batch[0].shape, np.mean(train_loss_list), np.mean(train_cross_entropy_list), l_r));
                train_loss_list = [];
                train_cross_entropy_list = [];

                val_batch = val_imdb_obj.generate_minibatch_region(region_batch_size);
                (val_summary, val_loss, val_cross_entropy) = sess.run(
                    [val_summary_op, model.loss, model.cross_entropy],
                    feed_dict={images: val_batch[0], labels: val_batch[1], keep_prob: 1.0});

                val_writer.add_summary(val_summary, global_step=i);
                val_loss_list.append(val_loss);
                val_cross_entropy_list.append(val_cross_entropy);

            if i % save_time == 0:
                val_loss_mean = np.mean(val_loss_list);
                val_cross_entropy_mean = np.mean(val_cross_entropy_list);

                print("************   validation set   ************");
                print("loss: {}, cls_cross_entropy: {}".format(val_loss_mean, val_cross_entropy_mean));
                val_loss_list = [];
                val_cross_entropy_list = [];

                val_loss_real = val_cross_entropy_mean;

                if val_loss_real < val_loss_best:
                    val_loss_best = val_loss_real;
                    model.save(model_dir_path, sess, saver);
                    print("Model Saved");