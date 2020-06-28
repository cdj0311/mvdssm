# coding:utf-8
import random
import tensorflow as tf
from tensorflow import feature_column as fc
import config
FLAGS = config.FLAGS


def build_user_model(features, mode, params):
    user_net = []
    user_inputs = params["feature_configs"].user_feature_columns
    with tf.variable_scope("user_side", partitioner=tf.fixed_size_partitioner(len(FLAGS.ps_hosts.split(",")), axis=0)):
        for key, value in user_inputs.items():
            input_fea = fc.input_layer(features, value)
            user_net.append(input_fea)
        user_net = tf.concat(user_net, axis=1)
        for idx, units in enumerate(params["hidden_units"]):
            user_net = tf.layers.dense(user_net, units=units, activation=tf.nn.leaky_relu, name="user_fc_layer_%s"%idx)
        user_net = tf.nn.l2_normalize(user_net)
    return user_net

def build_item_model(features, mode, params):
    item_net = []
    item_inputs = params["feature_configs"].item_feature_columns
    with tf.variable_scope("item_side", partitioner=tf.fixed_size_partitioner(len(FLAGS.ps_hosts.split(",")), axis=0)):
        for key, value in item_inputs.items():
            input_fea = fc.input_layer(features, value)
            item_net.append(input_fea)
        item_net = tf.concat(item_net, axis=1)
        for idx, units in enumerate(params["hidden_units"]):
            item_net = tf.layers.dense(item_net, units=units, activation=tf.nn.leaky_relu, name="item_fc_layer_%s"%idx)
        item_net = tf.nn.l2_normalize(item_net)
    return item_net

def model_fn(features, labels, mode, params):
    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        if FLAGS.export_user_model:
            user_encoder = build_user_model(features, mode, params)
            predictions = {"user_vector": user_encoder}
        elif FLAGS.export_item_model:
            item_encoder = build_item_model(features, mode, params)
            predictions = {"item_vector": item_encoder}
        export_outputs = {"predictions": tf.estimator.export.PredictOutput(outputs=predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    user_encoder = build_user_model(features, mode, params)
    item_encoder = build_item_model(features, mode, params)

    # 随机采样负样本
    with tf.name_scope("rotate"):
        tmp = tf.tile(item_encoder, [1, 1])
        item_encoder_fd = item_encoder
        for i in range(FLAGS.NEG):
            rand = tf.cast(((random.random() + i) * tf.cast(FLAGS.batch_size, tf.float32) / FLAGS.NEG), tf.int32)
            item_encoder_fd = tf.concat([item_encoder_fd,
                                      tf.slice(tmp, [rand, 0], [FLAGS.batch_size - rand, -1]),
                                      tf.slice(tmp, [0, 0], [rand, -1])], axis=0)
        user_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(user_encoder), axis=1, keepdims=True)),[FLAGS.NEG + 1, 1])
        item_norm = tf.sqrt(tf.reduce_sum(tf.square(item_encoder_fd), axis=1, keepdims=True))
        prod = tf.reduce_sum(tf.multiply(tf.tile(user_encoder, [FLAGS.NEG + 1, 1]), item_encoder_fd), axis=1,keepdims=True)
        norm_prod = tf.multiply(user_norm, item_norm)
        cos_sim_raw = tf.truediv(prod, norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [FLAGS.NEG + 1, -1])) * 20

    # 最大化正样本概率
    with tf.name_scope("loss"):
        prob = tf.nn.softmax(cos_sim)
        hit_prob = tf.slice(prob, [0, 0], [-1, 1])
        loss = -tf.reduce_mean(tf.log(hit_prob))
        correct_prediction = tf.cast(tf.equal(tf.argmax(prob, 1), 0), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    # Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={})

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(params["learning_rate"], global_step, 100000, 0.9, staircase=True)
        train_op = (tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
