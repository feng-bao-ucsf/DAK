import tensorflow as tf
from DAK_helper import DAK
import numpy as np


def train(batch_path_prefix, label_path_prefix, cov_path_prefix=None, p_val_path, batch_num, batch_size, pathway_num,
          max_path_len):
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    config_.allow_soft_placement = True
    sess = tf.Session(config=config_)

    with sess.as_default():
        dak = DAK(sess,
                  batch_path_prefix=batch_path_prefix, label_path_prefix=label_path_prefix,
                  cov_path_prefix=cov_path_prefix, p_val_path=p_val_path,
                  batch_num=batch_num, batch_size=batch_size, pathway_num=pathway_num, max_path_len=max_path_len)
        dak.train_DAK()


def one_hot_convert(geno):
    geno_one_hot = np.zeros([geno.shape[0], geno.shape[1], 3])

    for i in range(geno.shape[0]):
        geno_one_hot[i, np.arange(geno.shape[1]), geno[i, :]] = 1

    return geno_one_hot
