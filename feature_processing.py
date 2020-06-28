#encoding:utf-8
import os, json, codecs
import tensorflow as tf
from tensorflow import feature_column as fc
import config

FLAGS = config.FLAGS

class FeatureConfig(object):
    def __init__(self):
        self.user_feature_columns = dict()
        self.item_feature_columns = dict()
        self.all_columns = dict()
        self.feature_spec = dict()

    def create_features_columns(self):
        """
        features_def["user_classes"] = tf.io.FixedLenFeature([5], tf.int64)  # 用户兴趣
        features_def["user_age"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户年龄
        features_def["user_gender"] = tf.io.FixedLenFeature([1], tf.int64)  # 用户性别
        features_def["user_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # 用户向量
        features_def["item_classes"] = tf.io.FixedLenFeature([1], tf.int64)  # item分类
        features_def["item_vector"] = tf.io.FixedLenFeature([128], tf.float32)  # item向量
        """
        user_classes_embed = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="user_classes", 
                                                                                       hash_bucket_size=40, dtype=tf.int64),
                                                 dimension=64, combiner='mean', initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1, dtype=tf.float32)
                                                 )
        user_age_embed = fc.embedding_column(fc.categorical_column_with_identity(key="user_age", 
                                                                                 num_bucket=6, dtype=tf.int64),
                                                 dimension=8, combiner='mean', initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1, dtype=tf.float32)
                                                 )
        user_gender_embed = fc.embedding_column(fc.categorical_column_with_identity(key="user_gender", 
                                                                                    num_bucket=3, dtype=tf.int64),
                                                 dimension=8, combiner='mean', initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1, dtype=tf.float32)
                                                 )
                                                 
        item_classes_embed = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="item_classes", 
                                                                                        hash_bucket_size=40, dtype=tf.int64),
                                                 dimension=64, combiner='mean', initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1, dtype=tf.float32)
                                                 )
                                                 
        user_vector_input = fc.numeric_column(key="user_vector", shape=(128,), default_value=[0.0]*128, dtype=tf.float32)
        item_vector_input = fc.numeric_column(key="item_vector", shape=(128,), default_value=[0.0]*128, dtype=tf.float32)

        for key, value in self.user_feature_columns.items():
            self.all_columns[key] = value
        for key, value in self.item_feature_columns.items():
            self.all_columns[key] = value
            
        self.feature_spec = tf.feature_column.make_parse_example_spec(self.all_columns.values())
        return self
