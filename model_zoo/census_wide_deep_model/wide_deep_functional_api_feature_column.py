import itertools
import tensorflow as tf

from model_zoo.census_wide_deep_model.feature_config_gen import (
    FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY,
    INPUT_SCHEMAS,
    TRANSFORM_OUTPUTS
)
from model_zoo.census_wide_deep_model.feature_info_util import (
    FeatureTransformInfo,
    TransformOp,
)
from model_zoo.census_wide_deep_model.keras_process_layer import (
    AddIdOffset,
    CategoryHash,
    CategoryLookup,
    Group,
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

def get_input_layers_from_meta(input_schemas):
    input_layers = {}

    for schema_info in input_schemas:
        input_layers[schema_info.name] = tf.keras.layers.Input(
                name=schema_info.name, shape=(1,), dtype=schema_info.dtype
            )

    return input_layers

def transform_from_meta(inputs):
    transformed = inputs.copy()

    for feature_transform_info in FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY:
        if feature_transform_info.op_name == TransformOp.HASH:
            transformed[feature_transform_info.output_name] = CategoryHash(
                feature_transform_info.param
            )(transformed[feature_transform_info.input_name])
        elif feature_transform_info.op_name == TransformOp.BUCKETIZE:
            transformed[feature_transform_info.output_name] = NumericBucket(
                feature_transform_info.param
            )(transformed[feature_transform_info.input_name])
        elif feature_transform_info.op_name == TransformOp.LOOKUP:
            transformed[feature_transform_info.output_name] = CategoryLookup(
                feature_transform_info.param
            )(transformed[feature_transform_info.input_name])
        elif feature_transform_info.op_name == TransformOp.GROUP:
            group_inputs = [
                transformed[name] for name in feature_transform_info.input_name
            ]
            offsets = list(itertools.accumulate([0] + feature_transform_info.param[:-1]))
            transformed[feature_transform_info.output_name] = Group(None)(
                group_inputs
            )
        elif feature_transform_info.op_name == TransformOp.EMBEDDING:
            # The num_buckets should be calcualte from the group items
            group_identity = tf.feature_column.categorical_column_with_identity(
                feature_transform_info.input_name,
                num_buckets=feature_transform_info.param[0]
            )
            group_embedding = tf.feature_column.embedding_column(
                group_identity,
                dimension=feature_transform_info.param[1]
            )
            transformed[feature_transform_info.output_name] = tf.keras.layers.DenseFeatures(
                [group_embedding]
            )({
                feature_transform_info.input_name: transformed[feature_transform_info.input_name]
            })
        elif feature_transform_info.op_name == TransformOp.ARRAY:
            transformed[feature_transform_info.output_name] = [
                transformed[name] for name in feature_transform_info.input_name
            ]

    return tuple([transformed[name] for name in TRANSFORM_OUTPUTS])
    # return transformed["group1_embedding_deep"]
    # return transformed["group1"], transformed["group2"], transformed["group1_embedding_wide"]


def transform_code_gen(source_inputs):
    inputs = source_inputs.copy()

    education_hash = CategoryHash(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[0].param)(inputs["education"])
    occupation_hash = CategoryHash(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[1].param)(inputs["occupation"])
    native_country_hash = CategoryHash(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[2].param)(inputs["native_country"])
    workclass_lookup = CategoryLookup(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[3].param)(inputs["workclass"])
    marital_status_lookup = CategoryLookup(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[4].param)(inputs["marital_status"])
    relationship_lookup = CategoryLookup(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[5].param)(inputs["relationship"])
    race_lookup = CategoryLookup(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[6].param)(inputs["race"])
    sex_lookup = CategoryLookup(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[7].param)(inputs["sex"])
    age_bucketize = NumericBucket(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[8].param)(inputs["age"])
    capital_gain_bucketize = NumericBucket(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[9].param)(inputs["capital_gain"])
    capital_loss_bucketize = NumericBucket(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[10].param)(inputs["capital_loss"])
    hours_per_week_bucketize = NumericBucket(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[11].param)(inputs["hours_per_week"])

    group1 = Group(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[12].param)([workclass_lookup, hours_per_week_bucketize, capital_gain_bucketize, capital_loss_bucketize])
    group2 = Group(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[13].param)([education_hash, marital_status_lookup, relationship_lookup, occupation_hash])
    group3 = Group(FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[14].param)([age_bucketize, sex_lookup, race_lookup, native_country_hash])

    inputs["group1"] = group1
    inputs["group2"] = group2
    inputs["group3"] = group3

    group1_wide_embedding_column = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
                "group1",
                num_buckets=1000
            ),
        dimension=FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[15].param)
    # group1_embedding_wide = tf.keras.layers.DenseFeatures([group1_wide_embedding_column])({"group1": group1})
    group1_embedding_wide = tf.keras.layers.DenseFeatures([group1_wide_embedding_column])(inputs)
    inputs["group1_embedding_wide"] = group1_embedding_wide

    group2_deep_embedding_column = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(
                "group2",
                num_buckets=1000
            ),
        dimension=FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY[18].param
    )
    # group2_embedding_deep = tf.keras.layers.DenseFeatures([group2_deep_embedding_column])({"group2": group2})
    group2_embedding_deep = tf.keras.layers.DenseFeatures([group2_deep_embedding_column])(inputs)

    return group1_embedding_wide, group2_embedding_deep


def transform_model():
    input_layers = get_input_layers_from_meta(INPUT_SCHEMAS)
    # transformed_output = transform_from_meta(input_layers)

    '''
    wide_embedding, deep_embedding = transform_code_gen(input_layers)
    return tf.keras.Model(
        inputs=input_layers,
        outputs={
            "wide": wide_embedding,
            "deep": deep_embedding
        }
    )
    '''

    transformed = transform_from_meta(input_layers)

    return tf.keras.Model(
        inputs=input_layers,
        outputs=transformed
    )
     
    '''
    wide_embeddings, deep_embeddings = transform_from_meta(input_layers)

    flat_wide_embedding = tf.keras.layers.concatenate(wide_embeddings, axis=-1)
    flat_deep_embedding = tf.keras.layers.concatenate(deep_embeddings, axis=-1)
    wide_output = tf.keras.layers.Dense(1)(flat_wide_embedding)
    deep_output = tf.keras.layers.Dense(1)(flat_deep_embedding)

    return tf.keras.Model(
        inputs=input_layers,
        outputs=wide_output + deep_output
    )
    '''

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
    '''
    model = custom_model()
    print(model.summary())
    '''

    '''
    input_layers = get_input_layers_from_meta(INPUT_SCHEMAS)
    transformed_outputs = transform_from_meta(input_layers)
    print(transformed_outputs)
    '''

    model = transform_model()
    print(model.summary())
