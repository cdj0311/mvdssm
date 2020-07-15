#encoding:utf-8
import os, json, codecs
import tensorflow as tf
from tensorflow import feature_column as fc
import config

FLAGS = config.FLAGS

def parse_exp(example):
    features_def = dict()
    features_def["label"] = tf.io.FixedLenFeature([1], tf.int64)
    
    features_def["user_classes"] = tf.io.FixedLenFeature([5], tf.int64)  # 用户兴趣
    features_def["user_age"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户年龄
    features_def["user_gender"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户性别
    features_def["user_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # 用户向量
    features_def["item_classes"] = tf.io.FixedLenFeature([1], tf.int64)  # item分类
    features_def["item_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # item向量
    
    features = tf.io.parse_single_example(example, features_def)
    label = features["label"]
    del features["label"]
    return features, label


def train_input_fn(filenames=None,
                   batch_size=128,
                   shuffle_buffer_size=1000):
    with tf.gfile.Open(filenames) as f:
        filenames = f.read().split()
    
    if FLAGS.run_on_cluster:
        files_all = []
        for f in filenames:
            files_all += tf.gfile.Glob(f)
        train_worker_num = len(FLAGS.worker_hosts.split(","))
        hash_id = FLAGS.task_index if FLAGS.job_name == "worker" else train_worker_num - 1
        files_shard = [files for i, files in enumerate(files_all) if i % train_worker_num == hash_id]
        files = tf.data.Dataset.list_files(files_shard)
        dataset = files.apply(tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), 
                                                                  cycle_length=10,
                                                                  buffer_output_elements=batch_size*20,
                                                                  sloppy=True))
        #dataset = tf.data.TFRecordDataset(files_shard)
    else:
        files = tf.data.Dataset.list_files(filenames)
        dataset = files.apply(tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), 
                                                              buffer_output_elements=batch_size*4, 
                                                              cycle_length=4,
                                                              sloppy=True))
    #dataset = dataset.shuffle(batch_size*10)
    dataset = dataset.map(parse_exp, num_parallel_calls=8)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    return dataset

def eval_input_fn(filenames=None,
                  batch_size=128):
    with tf.gfile.Open(filenames) as f:
        filenames = f.read().split()
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.contrib.data.parallel_interleave(lambda filename: tf.data.TFRecordDataset(files), buffer_output_elements=batch_size*12, cycle_length=8))
    dataset = dataset.map(parse_exp, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


