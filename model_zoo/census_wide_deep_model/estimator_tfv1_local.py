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

import pandas as pd
import tensorflow as tf
from census_common import COLUMN_NAMES, MODEL_DIR, build_estimator

tf.logging.set_verbosity(tf.logging.INFO)


def get_input_pandas_data(file):
    census = pd.read_csv(file, header=None, skipinitialspace=True)
    census.columns = COLUMN_NAMES

    census["income_bracket"] = census["income_bracket"].apply(
        lambda label: 0 if label == " <=50K" else 1
    )

    labels = census.pop("income_bracket")
    features = census

    return features, labels


x_train, y_train = get_input_pandas_data("./data/census/raw/adult.data")
x_test, y_test = get_input_pandas_data("./data/census/raw/adult.test")

train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, batch_size=16, num_epochs=8, shuffle=True
)

eval_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=x_test, y=y_test, batch_size=16, num_epochs=1, shuffle=False
)

# Build train spec and eval spec
train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn,
    hooks=[
        tf.estimator.StepCounterHook(every_n_steps=100),
        tf.estimator.CheckpointSaverHook(
            checkpoint_dir=MODEL_DIR, save_steps=1000
        ),
    ],
)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

# Build estimator
estimator = build_estimator()

# Execute train and evaluate
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
