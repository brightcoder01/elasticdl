import itertools

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import feature_column as fc

from elasticdl.python.elasticdl.feature_column import feature_column as edl_fc
from model_zoo.census_model_sqlflow.feature_configs import (
    INPUT_SCHEMAS,
    LABEL_KEY,
    age_bucketize,
    capital_gain_bucketize,
    capital_loss_bucketize,
    education_hash,
    group1,
    group1_embedding_deep,
    group1_embedding_wide,
    group2,
    group2_embedding_deep,
    group2_embedding_wide,
    group3,
    group3_embedding_deep,
    hours_per_week_bucketize,
    marital_status_lookup,
    native_country_hash,
    occupation_hash,
    race_lookup,
    relationship_lookup,
    sex_lookup,
    workclass_lookup,
)
from model_zoo.census_model_sqlflow.keras_process_layers import (
    Concat,
    FingerPrint,
    Lookup,
    NumericBucket,
)


# The model definition from model zoo. It's functional style.
# Input Params:
#   input_layers: The input layers dict of feature inputs
#   input_tensors: list of integer tensors
def deepfm_classifier(
    input_layers, wide_feature_columns, deep_feature_columns
):
    # Wide Part
    wide_embeddings = []
    for wide_feature_column in wide_feature_columns:
        if not isinstance(wide_feature_column, list):
            wide_feature_column = [wide_feature_column]
        wide_embedding = tf.keras.layers.DenseFeatures(wide_feature_column)(
            input_layers
        )
        wide_embeddings.append(wide_embedding)

    deep_embeddings = []
    for deep_feature_column in deep_feature_columns:
        if not isinstance(deep_feature_column, list):
            deep_feature_column = [deep_feature_column]
        deep_embedding = tf.keras.layers.DenseFeatures(deep_feature_column)(
            input_layers
        )
        deep_embeddings.append(deep_embedding)

    group_num = len(deep_embeddings)
    embeddings = tf.concat(deep_embeddings, 1)  # shape = (None, group_num , 8)
    embeddings = tf.reshape(embeddings, shape=(-1, group_num, 8))
    emb_sum = K.sum(embeddings, axis=1)  # shape = (None, 8)
    emb_sum_square = K.square(emb_sum)  # shape = (None, 8)
    emb_square = K.square(embeddings)  # shape = (None, group_num, 8)
    emb_square_sum = K.sum(emb_square, axis=1)  # shape = (None, 8)
    second_order = 0.5 * tf.keras.layers.Subtract()(
        [emb_sum_square, emb_square_sum]
    )

    first_order = tf.keras.layers.Concatenate()(wide_embeddings)

    # Deep Part
    dnn_input = tf.keras.layers.Concatenate()(deep_embeddings)
    for i in [16, 8, 4]:
        dnn_input = tf.keras.layers.Dense(i)(dnn_input)

    # Output Part
    print(first_order)
    print(second_order)
    print(dnn_input)
    concat_input = tf.concat([first_order, second_order, dnn_input], 1)

    logits = tf.reduce_sum(concat_input, 1, keepdims=True)
    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))

    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="deepfm",
    )


# Build the input layers from the schema of the input features
def get_input_layers(input_schemas):
    input_layers = {}

    for schema_info in input_schemas:
        input_layers[schema_info.name] = tf.keras.layers.Input(
            name=schema_info.name, shape=(1,), dtype=schema_info.dtype
        )

    return input_layers


# It can be generated from the parsed meta in feature_configs using code_gen.
def transform(source_inputs):
    inputs = source_inputs.copy()

    education_hash_out = FingerPrint(education_hash.param)(inputs["education"])
    occupation_hash_out = FingerPrint(occupation_hash.param)(
        inputs["occupation"]
    )
    native_country_hash_out = FingerPrint(native_country_hash.param)(
        inputs["native_country"]
    )
    workclass_lookup_out = Lookup(workclass_lookup.param)(inputs["workclass"])
    marital_status_lookup_out = Lookup(marital_status_lookup.param)(
        inputs["marital_status"]
    )
    relationship_lookup_out = Lookup(relationship_lookup.param)(
        inputs["relationship"]
    )
    race_lookup_out = Lookup(race_lookup.param)(inputs["race"])
    sex_lookup_out = Lookup(sex_lookup.param)(inputs["sex"])
    age_bucketize_out = NumericBucket(age_bucketize.param)(inputs["age"])
    capital_gain_bucketize_out = NumericBucket(capital_gain_bucketize.param)(
        inputs["capital_gain"]
    )
    capital_loss_bucketize_out = NumericBucket(capital_loss_bucketize.param)(
        inputs["capital_loss"]
    )
    hours_per_week_bucketize_out = NumericBucket(
        hours_per_week_bucketize.param
    )(inputs["hours_per_week"])

    group1_offsets = list(itertools.accumulate([0] + group1.param[:-1]))
    group1_out = Concat(group1_offsets)(
        [
            workclass_lookup_out,
            hours_per_week_bucketize_out,
            capital_gain_bucketize_out,
            capital_loss_bucketize_out,
        ]
    )
    group2_offsets = list(itertools.accumulate([0] + group2.param[:-1]))
    group2_out = Concat(group2_offsets)(
        [
            education_hash_out,
            marital_status_lookup_out,
            relationship_lookup_out,
            occupation_hash_out,
        ]
    )
    group3_offsets = list(itertools.accumulate([0] + group3.param[:-1]))
    group3_out = Concat(group3_offsets)(
        [
            age_bucketize_out,
            sex_lookup_out,
            race_lookup_out,
            native_country_hash_out,
        ]
    )
    group_ids = [group1_out, group2_out, group3_out]
    group_max_ids = [sum(group1.param), sum(group2.param), sum(group3.param)]
    print("group_max_ids", group_max_ids)
    return group_ids, group_max_ids


# The following code has the same logic with the `transform` function above.
# It can be generated from the parsed meta in feature_configs using code_gen.
def transform_from_code_gen(source_inputs):
    education_hash_fc = fc.categorical_column_with_hash_bucket(
        "education", hash_bucket_size=education_hash.param
    )

    occupation_hash_fc = fc.categorical_column_with_hash_bucket(
        "occupation", hash_bucket_size=occupation_hash.param
    )

    native_country_hash_fc = fc.categorical_column_with_hash_bucket(
        "native_country", hash_bucket_size=native_country_hash.param
    )

    workclass_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "workclass", vocabulary_list=workclass_lookup.param
    )

    marital_status_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "marital_status", vocabulary_list=marital_status_lookup.param
    )

    relationship_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "relationship", vocabulary_list=relationship_lookup.param
    )

    race_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "race", vocabulary_list=race_lookup.param
    )

    sex_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "sex", vocabulary_list=sex_lookup.param
    )

    age_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("age"), boundaries=age_bucketize.param
    )

    capital_gain_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("capital_gain"),
        boundaries=capital_gain_bucketize.param,
    )

    capital_loss_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("capital_loss"),
        boundaries=capital_loss_bucketize.param,
    )

    hours_per_week_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("hours_per_week"),
        boundaries=hours_per_week_bucketize.param,
    )

    group1_fc = edl_fc.concat_column(
        categorical_columns=[
            workclass_lookup_fc,
            hours_per_week_bucketize_fc,
            capital_gain_bucketize_fc,
            capital_loss_bucketize_fc,
        ]
    )

    group2_fc = edl_fc.concat_column(
        categorical_columns=[
            education_hash_fc,
            marital_status_lookup_fc,
            relationship_lookup_fc,
            occupation_hash_fc,
        ]
    )

    group3_fc = edl_fc.concat_column(
        categorical_columns=[
            age_bucketize_fc,
            sex_lookup_fc,
            race_lookup_fc,
            native_country_hash_fc,
        ]
    )

    group1_wide_embedding_fc = fc.embedding_column(
        group1_fc, dimension=group1_embedding_wide.param[1],
    )

    group2_wide_embedding_fc = fc.embedding_column(
        group2_fc, dimension=group2_embedding_wide.param[1],
    )

    group1_deep_embedding_fc = fc.embedding_column(
        group1_fc, dimension=group1_embedding_deep.param[1],
    )

    group2_deep_embedding_fc = fc.embedding_column(
        group2_fc, dimension=group2_embedding_deep.param[1],
    )

    group3_deep_embedding_fc = fc.embedding_column(
        group3_fc, dimension=group3_embedding_deep.param[1],
    )

    wide_feature_columns = [
        [group1_wide_embedding_fc],
        [group2_wide_embedding_fc],
    ]

    deep_feature_columns = [
        [group1_deep_embedding_fc],
        [group2_deep_embedding_fc],
        [group3_deep_embedding_fc],
    ]

    return wide_feature_columns, deep_feature_columns


# The entry point of the submitter program
def custom_model():
    input_layers = get_input_layers(input_schemas=INPUT_SCHEMAS)
    wide_feature_columns, deep_feature_columns = transform_from_code_gen(
        input_layers
    )

    return deepfm_classifier(input_layers, wide_feature_columns, deep_feature_columns)


def loss(labels, predictions):
    logits = predictions["logits"]
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, (-1, 1)), tf.float32),
            logits=logits,
        )
    )


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def eval_metrics_fn():
    return {
        "logits": {
            "accuracy": lambda labels, predictions: tf.equal(
                tf.cast(tf.reshape(predictions, [-1]) > 0.5, tf.int32),
                tf.cast(tf.reshape(labels, [-1]), tf.int32),
            )
        },
        "probs": {"auc": tf.keras.metrics.AUC()},
    }


def learning_rate_scheduler(model_version):
    if model_version < 5000:
        return 0.0003
    elif model_version < 12000:
        return 0.0002
    else:
        return 0.0001


def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        feature_description = {}
        for schema in INPUT_SCHEMAS:
            feature_description[schema.name] = tf.io.FixedLenFeature(
                (1,), schema.dtype
            )
        feature_description[LABEL_KEY] = tf.io.FixedLenFeature([], tf.int64)

        print(feature_description)
        parsed_record = tf.io.parse_single_example(record, feature_description)
        label = parsed_record.pop(LABEL_KEY)

        return parsed_record, label

    dataset = dataset.map(_parse_data)

    return dataset


if __name__ == "__main__":
    model = custom_model()
    print(model.summary())

    output = model.call(
        {
            "education": tf.constant(["Bachelors"], tf.string),
            "occupation": tf.constant(["Tech-support"], tf.string),
            "native_country": tf.constant(["United-States"], tf.string),
            "workclass": tf.constant(["Private"], tf.string),
            "marital_status": tf.constant(["Separated"], tf.string),
            "relationship": tf.constant(["Husband"], tf.string),
            "race": tf.constant(["White"], tf.string),
            "sex": tf.constant(["Female"], tf.string),
            "age": tf.constant([18], tf.float32),
            "capital_gain": tf.constant([100.0], tf.float32),
            "capital_loss": tf.constant([1.0], tf.float32),
            "hours_per_week": tf.constant([40], tf.float32),
        }
    )

    print(output)
