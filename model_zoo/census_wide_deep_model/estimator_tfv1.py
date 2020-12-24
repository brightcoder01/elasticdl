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
from tensorflow import feature_column as fc

tf.logging.set_verbosity(tf.logging.INFO)

WORK_CLASS_VOCABULARY = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]

MARITAL_STATUS_VOCABULARY = [
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]

RELATIONSHIP_VOCABULARY = [
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]

RACE_VOCABULARY = [
    "White",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
    "Black",
]

SEX_VOCABULARY = ["Female", "Male"]

AGE_BOUNDARIES = [0, 20, 40, 60, 80]
CAPITAL_GAIN_BOUNDARIES = [6000, 6500, 7000, 7500, 8000]
CAPITAL_LOSS_BOUNDARIES = [2000, 2500, 3000, 3500, 4000]
HOURS_BOUNDARIES = [10, 20, 30, 40, 50, 60]


def get_linear_and_dnn_feature_columns():
    education_hash_fc = fc.categorical_column_with_hash_bucket(
        "education", hash_bucket_size=30
    )

    occupation_hash_fc = fc.categorical_column_with_hash_bucket(
        "occupation", hash_bucket_size=30
    )

    native_country_hash_fc = fc.categorical_column_with_hash_bucket(
        "native_country", hash_bucket_size=100
    )

    workclass_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "workclass", vocabulary_list=WORK_CLASS_VOCABULARY
    )

    marital_status_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "marital_status", vocabulary_list=MARITAL_STATUS_VOCABULARY
    )

    relationship_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "relationship", vocabulary_list=RELATIONSHIP_VOCABULARY
    )

    race_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "race", vocabulary_list=RACE_VOCABULARY
    )

    sex_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "sex", vocabulary_list=SEX_VOCABULARY
    )

    age_numeric_fc = fc.numeric_column("age")

    capital_gain_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("capital_gain"), boundaries=CAPITAL_GAIN_BOUNDARIES
    )

    capital_loss_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("capital_loss"), boundaries=CAPITAL_LOSS_BOUNDARIES,
    )

    hours_per_week_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("hours_per_week"), boundaries=HOURS_BOUNDARIES,
    )

    linear_columns = [
        age_numeric_fc,
        capital_gain_bucketize_fc,
        capital_loss_bucketize_fc,
        hours_per_week_bucketize_fc,
    ]

    EMBEDDING_DIMENSION = 8
    dnn_columns = [
        fc.embedding_column(education_hash_fc, EMBEDDING_DIMENSION),
        fc.embedding_column(occupation_hash_fc, EMBEDDING_DIMENSION),
        fc.embedding_column(native_country_hash_fc, EMBEDDING_DIMENSION),
        fc.embedding_column(workclass_lookup_fc, EMBEDDING_DIMENSION),
        fc.embedding_column(marital_status_lookup_fc, EMBEDDING_DIMENSION),
        fc.embedding_column(relationship_lookup_fc, EMBEDDING_DIMENSION),
        fc.embedding_column(race_lookup_fc, EMBEDDING_DIMENSION),
        fc.embedding_column(sex_lookup_fc, EMBEDDING_DIMENSION),
    ]

    return linear_columns, dnn_columns


__COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]


def get_input_pandas_data(file):
    census = pd.read_csv(file, header=None, skipinitialspace=True)
    census.columns = __COLUMN_NAMES

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

model_dir = "./models/census"

# Build estimator
linear_columns, dnn_columns = get_linear_and_dnn_feature_columns()
estimator = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=linear_columns,
    dnn_feature_columns=dnn_columns,
    dnn_hidden_units=[256, 128],
    n_classes=2,
)

# Build train spec and eval spec
train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn,
    hooks=[
        tf.estimator.StepCounterHook(every_n_steps=100),
        tf.estimator.CheckpointSaverHook(
            checkpoint_dir=model_dir, save_steps=1000
        ),
    ],
)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

# Execute train and evaluate
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
