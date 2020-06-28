# encoding:utf-8
import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as fc
import data_inputs
from feature_processing import FeatureConfig
import model
import config

FLAGS = config.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuid

if FLAGS.run_on_cluster:
    cluster = json.loads(os.environ["TF_CONFIG"])
    task_index = int(os.environ["TF_INDEX"])
    task_type = os.environ["TF_ROLE"]


def main(unused_argv):
    feature_configs = FeatureConfig().create_features_columns()
    classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                                                      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                                                      keep_checkpoint_max=3),
                                        params={"feature_configs": feature_configs,
                                                "hidden_units": list(map(int, FLAGS.hidden_units.split(","))),
                                                "learning_rate": FLAGS.learning_rate}
                                        )
    def train_eval_model():
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: data_inputs.train_input_fn(FLAGS.train_data, FLAGS.batch_size),
                                            max_steps=FLAGS.train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data_inputs.eval_input_fn(FLAGS.eval_data, FLAGS.batch_size),
                                          start_delay_secs=60,
                                          throttle_secs = 30,
                                          steps=1000)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    def train_model():
        from tensorflow.python import debug as tf_debug
        debug_hook = tf_debug.LocalCLIDebugHook()
        classifier.train(input_fn=lambda: fe.train_input_fn(FLAGS.train_data, FLAGS.batch_size), steps=1000, hooks=[debug_hook,])

    def eval_model():
        classifier.evaluate(input_fn=lambda: fe.eval_input_fn(FLAGS.eval_data, FLAGS.batch_size), steps=1000)

    if FLAGS.is_eval:
        eval_model()

    if FLAGS.train_eval:
        train_eval_model()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
