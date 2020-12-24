# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from census_common import (
    CATEGORICAL_FEATURE_KEYS,
    LABEL_KEY,
    MODEL_DIR,
    NUMERIC_FEATURE_KEYS,
    build_estimator,
)

from elasticai_api.common.data_shard_service import build_data_shard_service
from elasticai_api.proto import elasticai_api_pb2
from elasticai_api.tensorflow.recordio_reader import RecordIODataReader

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 32

data_shard_service = build_data_shard_service(batch_size=BATCH_SIZE)


def get_input_fn(data_dir, train=True):
    def _parse_data(record):
        feature_description = dict(
            [
                (name, tf.io.FixedLenFeature((1,), tf.string))
                for name in CATEGORICAL_FEATURE_KEYS
            ]
            + [
                (name, tf.io.FixedLenFeature((1,), tf.float32))
                for name in NUMERIC_FEATURE_KEYS
            ]
            + [(LABEL_KEY, tf.io.FixedLenFeature([], tf.int64))]
        )

        parsed_record = tf.io.parse_single_example(record, feature_description)
        label = parsed_record.pop(LABEL_KEY)

        return parsed_record, label

    def _input_fn():
        data_reader = RecordIODataReader(data_dir=data_dir)

        def _gen():
            while True:
                task = data_shard_service.get_task()
                if task.type != elasticai_api_pb2.TRAINING:
                    break

                for data in data_reader.read_records(task):
                    if data:
                        yield data

        dataset = tf.data.Dataset.from_generator(
            _gen, data_reader.records_output_types
        )

        dataset = dataset.map(_parse_data).batch(BATCH_SIZE)
        if train:
            dataset = dataset.shuffle(buffer_size=1024)

        return dataset

    return _input_fn


estimator = build_estimator()


class ReportBatchHook(tf.train.SessionRunHook):
    def __init__(self, data_shard_service):
        self._data_shard_service = data_shard_service

    def after_run(self, run_context, run_values):
        self._data_shard_service.report_batch_done()


# Build train spec and eval spec
train_spec = tf.estimator.TrainSpec(
    input_fn=get_input_fn("/data/census/train", True),
    hooks=[
        tf.estimator.StepCounterHook(every_n_steps=100),
        tf.estimator.CheckpointSaverHook(
            checkpoint_dir=MODEL_DIR, save_steps=1000
        ),
        ReportBatchHook(data_shard_service),
    ],
)
eval_spec = tf.estimator.EvalSpec(
    input_fn=get_input_fn("/data/census/train", False)
)

# Execute train and evaluate
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
