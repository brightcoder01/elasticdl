import tensorflow as tf

from model_zoo.census_wide_deep_model.feature_config import (
    CATEGORICAL_FEATURE_KEYS,
    FEATURE_GROUPS,
    LABEL_KEY,
    MODEL_INPUTS,
    NUMERIC_FEATURE_KEYS,
    TransformOp,
)
from model_zoo.census_wide_deep_model.keras_process_layer import (
    AddIdOffset,
    NumericBucket,
)


# Auto generated `Input Layers` from SQLFlow statement
def get_input_layers(feature_groups):
    input_layers = {}
    for feature_group in feature_groups.values():
        for feature_info in feature_group:
            input_layers[feature_info.name] = tf.keras.layers.Input(
                name=feature_info.name, shape=(1,), dtype=feature_info.dtype
            )
    return input_layers


def transform_group(inputs, feature_group):
    """Transform the inputs and concatenate inputs in a group
    to a dense tensor
    """
    group_items = []
    for feature_info in feature_group:
        layer = get_transform_layer(feature_info)
        transform_output = layer(inputs[feature_info.name])
        group_items.append(transform_output)

    id_offsets = get_id_boundaries(feature_group)

    if id_offsets is not None:
        group_items = AddIdOffset(id_offsets[0:-1])(group_items)
    group_stack = tf.keras.layers.concatenate(group_items, axis=-1)
    return group_stack


# Auto generated `Transform Code` using code_gen from SQLFlow statement
def transform(inputs, feature_groups):
    outputs = inputs.copy()

    wide_embeddings = []
    deep_embeddings = []

    for group_name, feature_group in feature_groups.items():
        outputs[group_name] = transform_group(inputs, feature_group)

    return wide_embeddings, deep_embeddings


# The model definition in model zoo
def wide_deep_model(input_layers, wide_embeddings, deep_embeddings):
    # Wide Part
    wide = tf.keras.layers.Concatenate()(wide_embeddings)  # shape = (None, 3)

    # Deep Part
    dnn_input = tf.reshape(deep_embeddings, shape=(-1, 3 * 8))
    for i in [16, 8, 4]:
        dnn_input = tf.keras.layers.Dense(i)(dnn_input)

    # Output Part
    concat_input = tf.concat([wide, dnn_input], 1)

    logits = tf.reduce_sum(concat_input, 1, keepdims=True)
    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))

    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="wide_deep",
    )


# The code in submitter program
def custom_model():
    input_layers = get_input_layers(feature_groups=FEATURE_GROUPS)
    wide_embeddings, deep_embeddings = transform(
        input_layers, feature_groups=FEATURE_GROUPS
    )

    return wide_deep_model(input_layers, wide_embeddings, deep_embeddings)


if __name__ == "__main__":
    model = custom_model()
    print(model.summary())
