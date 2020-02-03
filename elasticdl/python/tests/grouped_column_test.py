import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.elasticdl.feature_column import feature_column


def call_feature_columns(feature_columns, input):
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    return dense_features(input)


class GroupedColumnTest(unittest.TestCase):
    def test_call_crossed_column(self):
        user_id = tf.feature_column.categorical_column_with_identity(
            "user_id", num_buckets=32
        )

        item_id = tf.feature_column.categorical_column_with_identity(
            "item_id", num_buckets=128)
        
        item_id_user_id_crossed = tf.feature_column.crossed_column([user_id, item_id], hash_bucket_size=200)

        crossed_indicator = tf.feature_column.indicator_column(item_id_user_id_crossed)

        output = call_feature_columns([crossed_indicator], 
        {
            "user_id": [10, 20],
            "item_id": [1, 100]
        })

        print(output)
        self.assertIsNotNone(output)

    def test_call_grouped_column(self):
        user_id = tf.feature_column.categorical_column_with_identity(
            "user_id", num_buckets=32
        )

        item_id = tf.feature_column.categorical_column_with_identity(
            "item_id", num_buckets=128)
        
        item_id_user_id_grouped = feature_column.grouped_column([user_id, item_id])

        grouped_indicator = tf.feature_column.indicator_column(item_id_user_id_grouped)
        # grouped_indicator = tf.feature_column.embedding_column(item_id_user_id_grouped, dimension=8)

        output = call_feature_columns([grouped_indicator], 
        {
            "user_id": [10, 20],
            "item_id": [1, 100]
        })

        self.assertIsNotNone(output)

if __name__ == "__main__":
    unittest.main()
