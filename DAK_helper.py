"""
Model construction of Deep Association Kernel
2019-9-6
"""

import tensorflow as tf
import numpy as np
import time
from scipy.stats import chi2
import os

# Network Structure Setting, refer to Supplementary Fig 1
SEQ_FORMAT = 3
CONV_WINDOW_SIZE = 100
CONV_KERNEL_NUM = 64
CONV_STEP_SIZE = 20  # related to the minimum set length

FC1_KERNEL_NUM = 32
FC2_KERNEL_NUM = 16

INITIAL_TIME = 10


class DAK(object):
    def __init__(self,
                 sess,
                 mean_max_weight=0.5,
                 learning_rate=1e-20,
                 training_iter=20,
                 display_step=5,
                 keep_prob=1,
                 batch_path_prefix='./',
                 label_path_prefix='./',
                 p_val_path='./',
                 cov_path_prefix=None,
                 batch_num=100,
                 batch_size=100,
                 pathway_num=183,
                 max_path_len=5000):
        self.sess = sess
        self.mean_max_weight = mean_max_weight  # the weight of max/mean
        self.learning_rate = learning_rate
        self.display_step = display_step

        self.training_iter = training_iter
        # define the batch index in training.
        # Set for simulation.
        self.batch_idx = np.random.permutation(batch_num)
        self.keep_prob = keep_prob
        self.batch_path_prefix = batch_path_prefix
        self.label_path_prefix = label_path_prefix
        self.p_val_path = p_val_path
        self.BIN_SIZE = max_path_len
        self.PATHWAY_NUM = pathway_num
        self.BATCH_SIZE = batch_size
        self.INDIVIDUAL_SIZE = batch_num * self.BATCH_SIZE
        self.POOLING_SIZE = self.BIN_SIZE / CONV_STEP_SIZE + int(self.BIN_SIZE % CONV_STEP_SIZE > 0)
        self.cov_path_prefix = cov_path_prefix
    
        
        self.construct_graph()
        

    def construct_graph(self):
        # Input all the data together once. As only part of the data was used,
        # the RAM size should be enough to load all the data at once.
        self.inputs = tf.placeholder(
            tf.float32, [self.PATHWAY_NUM, self.BATCH_SIZE, self.BIN_SIZE, SEQ_FORMAT])
        # for each pathway, give an independent label vector.
        self.label = tf.placeholder(
            tf.float32, [self.BATCH_SIZE, 2])

        # Convolution layer:
        # Output: [self.PATHWAY_NUM, self.INDIVIDUAL_SIZE,
        # ceil(MAX_PAHTWAY_SIZE/CONV_STEP_SIZE), CONV_KERNEL_NUM]
        self.conv_weights = tf.Variable(
            tf.truncated_normal([1, CONV_WINDOW_SIZE,
                                 SEQ_FORMAT, CONV_KERNEL_NUM],
                                stddev=0.1))
        self.conv_bias = tf.Variable(tf.zeros([CONV_KERNEL_NUM]))
        self.conv_stride = [1, 1, CONV_STEP_SIZE, 1]
        self.conv_output = self.conv_rect(
            self.inputs, self.conv_weights, self.conv_bias, self.conv_stride)

        # Pooling layer:
        # Output: [self.PATHWAY_NUM, self.INDIVIDUAL_SIZE, 1, CONV_KERNEL_NUM]
        self.pooling_kernel = [1, 1, self.POOLING_SIZE, 1]
        self.pooling_stride = [1, 1, self.POOLING_SIZE, 1]
        self.pooling_output = self.pool_max(
            self.conv_output, self.pooling_kernel, self.pooling_stride)

        # Full connection layer 1:
        # Output:[self.PATHWAY_NUM, self.INDIVIDUAL_SIZE, 1,FC1_KERNEL_NUM]
        # using convolution to mimic the full connection
        # under each pathway set.
        self.fc_kernel_1 = tf.Variable(
            tf.truncated_normal([1, 1, CONV_KERNEL_NUM,
                                 FC1_KERNEL_NUM], stddev=0.1))
        self.fc_kernel_1_drop = tf.nn.dropout(
            self.fc_kernel_1, keep_prob=self.keep_prob)
        self.fc_bias_1 = tf.Variable(tf.zeros([FC1_KERNEL_NUM]))
        self.fc_stride_1 = [1, 1, 1, 1]
        self.fc_output_1 = self.conv_rect(
            self.pooling_output, self.fc_kernel_1_drop,
            self.fc_bias_1, self.fc_stride_1)

        # Full connection layer 2:
        # Output:[self.PATHWAY_NUM, self.INDIVIDUAL_SIZE, 1,FC2_KERNEL_NUM]
        self.fc_kernel_2 = tf.Variable(
            tf.truncated_normal([1, 1, FC1_KERNEL_NUM,
                                 FC2_KERNEL_NUM], stddev=0.1))
        self.fc_bias_2 = tf.Variable(tf.zeros([FC2_KERNEL_NUM]))
        self.fc_stride_2 = [1, 1, 1, 1]
        self.fc_output_2_ = self.conv_sigmoid(
            self.fc_output_1, self.fc_kernel_2,
            self.fc_bias_2, self.fc_stride_2)
        self.fc_output_2 = tf.squeeze(self.fc_output_2_)

        self.kernel_tensor_ = self.kernel_linear(self.fc_output_2)  # [S,N,N]

        # output layer
        # output: [S, N, 1, 2]
        self.fc_kernel_3 = tf.Variable(tf.truncated_normal([1, 1, self.BATCH_SIZE, 2]))
        self.fc_bias_3 = tf.Variable(tf.zeros([2]))
        self.fc_stride_3 = [1, 1, 1, 1]
        self.kernel_tensor = tf.expand_dims(
            self.kernel_tensor_, 2)  # [S,N,1,N]

        self.logit = self.conv_rect_no_relu(self.kernel_tensor, self.fc_kernel_3,
                                            self.fc_bias_3, self.fc_stride_3)

        # multiple instance layer
        # the maximum response represents the best association with label
        self.logit_max = tf.reduce_max(self.logit, 0)  # [N,1,2]
        self.logit_mean = tf.reduce_mean(self.logit, 0)
        self.logit_overall_ = tf.add(self.logit_max, tf.scalar_mul(
            self.mean_max_weight, self.logit_mean))
        self.logit_overall = tf.squeeze(self.logit_overall_)  # [N,2]

        if not self.cov_path_prefix == None:
            cov_buff = np.load(self.cov_path_prefix + '/batch_1.npy')
            self.cov = tf.placeholder(
                tf.float32, [self.BATCH_SIZE, cov_buff.shape[1]])
            self.cov_weight = tf.Variable(tf.truncated_normal([cov_buff.shape[1], 2]))
            self.cov_bias = tf.Variable(tf.zeros([2]))
            self.cov_regress = tf.add(tf.matmul(self.cov, self.cov_weight), self.cov_bias)
            self.logit_overall = self.logit_overall + self.cov_regress

        # loss
        self.cross_ent = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logit_overall, labels=self.label)
        self.l2_regular = tf.nn.l2_loss(self.fc_kernel_3)
        self.loss_overall = tf.reduce_mean(
            self.cross_ent + 0.1 * self.l2_regular)

        # accuracy
        correct_prediction = tf.equal(
            tf.argmax(self.logit_overall, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        case_tpr = tf.equal(
            tf.argmax(self.logit_overall, 1), tf.ones(self.BATCH_SIZE, dtype=tf.int64))
        self.case_pred = tf.reduce_mean(tf.cast(case_tpr, tf.float32))

    

    def train_DAK(self):
        print('P value path: ' + self.p_val_path)
        print('Data path: ' + self.batch_path_prefix)
        learn_rate_pl = tf.placeholder(tf.float32)
        print('Learning rate: ' + str(self.learning_rate))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learn_rate_pl).minimize(self.loss_overall)



        start_time = time.time()

        p_val_saver = np.ones([self.PATHWAY_NUM])
        saver_path = './saver_temp' + \
                     str(np.random.randint(1, 1000)) + '.ckpt'

        p_val_saver_pre = p_val_saver

        # if cov exist, train null model
        if not self.cov_path_prefix == None:
            print('Train model under null hypothesis')
            self.null_model()
            optimizer_null = tf.train.AdamOptimizer(
                learning_rate=learn_rate_pl).minimize(self.loss_overall_null)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            saver_null = tf.train.Saver({'cov_weight_null': self.cov_weight_null,
                                         'cov_bias_null': self.cov_bias_null})

            for train_iter in range(0, self.training_iter):
                for batch_iter in self.batch_idx:
                    seq_batch, seq_batch_label, seq_batch_cov = self.data_seq_next_batch_cov(
                        batch_iter)
                    
                    seq_batch_label = seq_batch_label[:, -1]
                    self.sess.run(optimizer_null,
                                  feed_dict={self.label_null: seq_batch_label,
                                             self.cov_null: seq_batch_cov,
                                             learn_rate_pl: self.learning_rate})
            saver_null.save(self.sess, 'null_saver.ckpt')
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver({'conv_weights': self.conv_weights,
                                'conv_bias': self.conv_bias,
                                'fc_kernel_1': self.fc_kernel_1,
                                'fc_bias_1': self.fc_bias_1,
                                'fc_kernel_2': self.fc_kernel_2,
                                'fc_bias_2': self.fc_bias_2})
        saver.save(self.sess, saver_path)
        for i in range(INITIAL_TIME):
            self.sess.run(init)
            for batch_iter in self.batch_idx:
                if self.cov_path_prefix == None:
                    seq_batch, seq_batch_label = self.data_seq_next_batch(
                        batch_iter)
                    self.sess.run(optimizer,
                                  feed_dict={self.inputs: seq_batch,
                                             self.label: seq_batch_label,
                                             learn_rate_pl: self.learning_rate})
                else:
                    seq_batch, seq_batch_label, seq_batch_cov = self.data_seq_next_batch_cov(
                        batch_iter)
                    self.sess.run(optimizer,
                                  feed_dict={self.inputs: seq_batch,
                                             self.label: seq_batch_label,
                                             self.cov: seq_batch_cov,
                                             learn_rate_pl: self.learning_rate})
            feature_saver = np.zeros([self.PATHWAY_NUM, self.INDIVIDUAL_SIZE, FC2_KERNEL_NUM])
            label_saver = np.zeros(self.INDIVIDUAL_SIZE)
            p_val_saver = np.ones([self.PATHWAY_NUM])
            start_loc = 0
            for batch_iter in self.batch_idx:
                if self.cov_path_prefix == None:
                    seq, label = self.data_seq_next_batch(batch_iter)
                    fc_output_2 = self.fc_output_2.eval(feed_dict={self.inputs: seq})
                else:
                    seq, label, cov = self.data_seq_next_batch_cov(batch_iter)
                    fc_output_2 = self.fc_output_2.eval(feed_dict={self.inputs: seq,
                                                                   self.cov: cov})
                deep_feature = fc_output_2  # [path, 100, 16]
                feature_saver[:, start_loc * self.BATCH_SIZE:(start_loc + 1) * self.BATCH_SIZE, :] = deep_feature
                
                label_saver[start_loc * self.BATCH_SIZE:(start_loc + 1) * self.BATCH_SIZE] = label[:, 1]
                
                start_loc = start_loc + 1
            # score test
            for path_iter in range(self.PATHWAY_NUM):
                kernel = self.sess.run(self.kernel_linear(feature_saver[path_iter, :, :]))
                p_val = self.score_test_(kernel, label_saver)
                p_val_saver[path_iter] = p_val
                # print('Test pathway %d' % path_iter)
                # print('P value is %f' % p_val)

            # method: only consider the top ten smallest p values
            pre_p_sort = np.sort(p_val_saver_pre)
            p_sort = np.sort(p_val_saver)
            pre_log_10 = np.log10(pre_p_sort[:10])
            cur_log_10 = np.log10(p_sort[:10])

            if np.sum(cur_log_10) < np.sum(pre_log_10):
                saver.save(self.sess, saver_path)
                p_val_saver_pre = p_val_saver
                print('saved!')
            print('Initial iteration: ' + str(i))

        print('Select best initial parameter.')
        saver.restore(self.sess, saver_path)
        os.remove(saver_path + '.meta')
        os.remove(saver_path + '.index')
        os.remove(saver_path + '.data-00000-of-00001')
        ######################################

        for train_iter in range(0, self.training_iter):
            print('**** iteration ' + str(train_iter) + ' ****')
            for batch_iter in self.batch_idx:
                if self.cov_path_prefix == None:
                    seq_batch, seq_batch_label = self.data_seq_next_batch(
                        batch_iter)
                    self.sess.run(optimizer,
                                  feed_dict={self.inputs: seq_batch,
                                             self.label: seq_batch_label,
                                             learn_rate_pl: self.learning_rate})
                else:
                    seq_batch, seq_batch_label, seq_batch_cov = self.data_seq_next_batch_cov(
                        batch_iter)
                    self.sess.run(optimizer,
                                  feed_dict={self.inputs: seq_batch,
                                             self.label: seq_batch_label,
                                             self.cov: seq_batch_cov,
                                             learn_rate_pl: self.learning_rate})

                if batch_iter % self.display_step == 0:
                    # print('**** iteration ' + str(train_iter) + ' ****')
                    if self.cov_path_prefix == None:
                        losses = self.loss_overall.eval(
                            feed_dict={self.inputs: seq_batch,
                                       self.label: seq_batch_label})
                    else:
                        losses = self.loss_overall.eval(
                            feed_dict={self.inputs: seq_batch,
                                       self.label: seq_batch_label,
                                       self.cov: seq_batch_cov})
                    print("Loss %f" % losses)
                    print("Time used: " + str(time.time() - start_time))

                    feature_saver = np.zeros([self.PATHWAY_NUM, self.INDIVIDUAL_SIZE, FC2_KERNEL_NUM])
                    label_saver = np.zeros(self.INDIVIDUAL_SIZE)
                    p_val_saver = np.ones([self.PATHWAY_NUM])
                    start_loc = 0
                    for batch_iter in self.batch_idx:
                        if self.cov_path_prefix == None:
                            seq, label = self.data_seq_next_batch(batch_iter)
                            fc_output_2 = self.fc_output_2.eval(feed_dict={self.inputs: seq})
                        else:
                            seq, label, cov = self.data_seq_next_batch_cov(batch_iter)
                            fc_output_2 = self.fc_output_2.eval(feed_dict={self.inputs: seq,
                                                                           self.cov: cov})

                        deep_feature = fc_output_2  # [path, 100, 16]
                        feature_saver[:, start_loc * self.BATCH_SIZE:(start_loc + 1) * self.BATCH_SIZE, :] \
                            = deep_feature
                        
                        label_saver[start_loc * self.BATCH_SIZE:(start_loc + 1) * self.BATCH_SIZE] = label[:, 1]
                        
                        start_loc = start_loc + 1
                    # score test
                    for path_iter in range(self.PATHWAY_NUM):
                        kernel = self.sess.run(self.kernel_linear(feature_saver[path_iter, :, :]))
                    p_val = self.score_test_(kernel, label_saver)
                    p_val_saver[path_iter] = p_val
                    np.savetxt(self.p_val_path, p_val_saver)
                    print("Time used: " + str(time.time() - start_time))

    def data_seq_next_batch(self, batch_idx):
        # load the batch sequence
        batch = np.load(self.batch_path_prefix + '/batch_' + str(batch_idx) +
                        '.npy')
        label = np.load(self.label_path_prefix + '/batch_' + str(batch_idx) +
                        '.npy')
 
        # label should be in one-hot format
        label_one_hot = np.zeros((len(label), 2))
        label_one_hot[range(len(label)), label] = 1
    
        return batch, label_one_hot

    def data_seq_next_batch_cov(self, batch_idx):
        # load the batch sequence
        batch = np.load(self.batch_path_prefix + '/batch_' + str(batch_idx) +
                        '.npy')
        label = np.load(self.label_path_prefix + '/batch_' + str(batch_idx) +
                        '.npy')
        cov = np.load(self.cov_path_prefix + '/batch_' + str(batch_idx) +
                      '.npy')
        
            # label should be in one-hot format
        label_one_hot = np.zeros((len(label), 2))
        label_one_hot[range(len(label)), label] = 1
        
        return batch, label_one_hot, cov

    def conv_rect(self, x, W, b, strides):
        # x: tensor with s*n*max_len*4
        # W: kernels for convolution
        # b: biases for each kernel
        x = tf.nn.conv2d(x, W, strides, padding='SAME')
        x = tf.nn.bias_add(x, b)  # for each kernel adding a bias value
        return tf.nn.relu(x)

    def conv_sigmoid(self, x, W, b, strides):
        # x: tensor with s*n*max_len*4
        # W: kernels for convolution
        # b: biases for each kernel
        x = tf.nn.conv2d(x, W, strides, padding='SAME')
        x = tf.nn.bias_add(x, b)  # for each kernel adding a bias value
        return tf.nn.sigmoid(x)

    def conv_rect_no_relu(self, x, W, b, strides):
        x = tf.nn.conv2d(x, W, strides, padding='SAME')
        x = tf.nn.bias_add(x, b)  # for each kernel adding a bias value
        return x

    def pool_max(self, x, ksize, strides):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')

    def pool_ave(self, x, ksize, strides):
        return tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding='SAME')

    def kernel_linear(self, x):
        return tf.matmul(x, x, transpose_b=True)

    def score_test_(self, kernel, label_all):
        # check if the label has the dtype of float
        # label_null = np.mean(label_all) * np.ones_like(label_all)
        label_null = self.label_null_hypo(label_all)
        label_centered = label_all - label_null
        label_centered = np.expand_dims(label_centered, 1)  # to [N, 1]

        K = np.matrix(kernel)
        label_centered = np.matrix(label_centered)

        Q_stat = label_centered.T * K * label_centered / 2
        D_0 = np.diag(label_null * (1 - label_null))
        X = np.ones_like(label_centered)

        D_0 = np.matrix(D_0)
        X = np.matrix(X)

        P_0 = D_0 - \
              D_0 * X * np.linalg.inv(X.T * D_0 * X) * X.T * D_0

        P_0 = np.matrix(P_0)

        mu_Q = np.trace(P_0 * K) / 2 + np.spacing(1)
        I_tt = np.trace(K * P_0 * K * P_0) / 2
        I_ts = np.trace(P_0 * K * P_0) / 2
        I_ss = np.trace(P_0 * P_0.T) / 2 + np.spacing(1)
        I_tt_expand = I_tt - np.square(I_ts) / I_ss + np.spacing(1)
        k = I_tt_expand / (2 * mu_Q)
        v = 2 * np.square(mu_Q) / I_tt_expand
        # print(mu_Q)
        p_val = 1 - chi2.cdf(Q_stat / k, v)
        return p_val[0, 0]

    def label_null_hypo(self, label_all):
        if self.cov_path_prefix == None:
            # no covariance existed
            label_null = np.mean(label_all) * np.ones_like(label_all)
        else:
            saver_null = tf.train.Saver({'cov_weight_null': self.cov_weight_null,
                                         'cov_bias_null': self.cov_bias_null})
            saver_null.restore(self.sess, 'null_saver.ckpt')
            label_null = np.zeros([self.INDIVIDUAL_SIZE])
            start_loc = 0
            for batch_iter in self.batch_idx:
                seq, label, cov = self.data_seq_next_batch_cov(batch_iter)
                
                label = label[:, 1]
                label_null_temp = self.cov_regress_null.eval(
                    feed_dict={self.cov_null: cov,
                               self.label_null: label})
                label_null[start_loc * self.BATCH_SIZE:(start_loc + 1) * self.BATCH_SIZE] = label_null_temp
                start_loc = start_loc + 1
        return label_null

    def null_model(self):

        cov = np.load(self.cov_path_prefix + '/batch_1.npy')
        self.cov_null = tf.placeholder(
            tf.float32, [self.BATCH_SIZE, cov.shape[1]])
        self.label_null = tf.placeholder(
            tf.float32, [self.BATCH_SIZE])
        self.cov_weight_null = tf.Variable(tf.truncated_normal([cov.shape[1], 1]))
        self.cov_bias_null = tf.Variable(tf.zeros([1]))
        self.cov_regress_null_ = tf.add(tf.matmul(self.cov_null, self.cov_weight_null), self.cov_bias_null)
        self.cov_regress_null_ = tf.squeeze(self.cov_regress_null_)

        # loss
        
        self.cov_regress_null = tf.math.sigmoid(self.cov_regress_null_)
        self.loss_overall_null = tf.reduce_mean(
            -tf.reduce_sum(self.label_null * tf.log(self.cov_regress_null)))
