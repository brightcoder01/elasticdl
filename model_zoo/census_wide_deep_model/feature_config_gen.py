import tensorflow as tf

from model_zoo.census_wide_deep_model.feature_info_util import (
    FeatureInfo,
    FeatureTransformInfo,
    TransformOp,
)

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

RELATION_SHIP_VOCABULARY = [
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

TRANSFORM_OUTPUT = {
    "wide_embeddings": ["group1_wide_embedding", "group2_wide_embedding"],
    "deep_embeddings": [
        "group1_deep_embedding",
        "group2_deep_embedding",
        "group3_deep_embedding",
    ],
}

education_hash = FeatureTransformInfo("education_hash", ["education"], "education_hash", TransformOp.HASH, tf.string, 30)
occupation_hash = FeatureTransformInfo("occupation_hash", ["occupation"], "occupation_hash", TransformOp.HASH, tf.string, 30)
native_country_hash = FeatureTransformInfo("native_country_hash", "native_country", "native_country_hash", TransformOp.HASH, tf.string, 100)

workclass_lookup = FeatureTransformInfo("workclass_lookup", "workclass", "workclass_lookup", TransformOp.LOOKUP, tf.string, WORK_CLASS_VOCABULARY)
marital_status_lookup = FeatureTransformInfo("marital_status_lookup", "marital_status", "marital_status_lookup", TransformOp.LOOKUP, tf.string, MARITAL_STATUS_VOCABULARY)
relationship_lookup = FeatureTransformInfo("relationship_lookup", "relationship", "relationship_lookup", TransformOp.LOOKUP, tf.string, RELATION_SHIP_VOCABULARY)
race_lookup = FeatureTransformInfo("race_lookup", "race", "race_lookup", TransformOp.LOOKUP, tf.string, RACE_VOCABULARY)
sex_lookup = FeatureTransformInfo("sex_lookup", "sex", "sex_lookup", TransformOp.LOOKUP, tf.string, SEX_VOCABULARY)

age_bucketize = FeatureTransformInfo("age_bucketize", "age", "age_bucketize", TransformOp.BUCKETIZE, tf.float32, AGE_BOUNDARIES)
capital_gain_bucketize = FeatureTransformInfo("capital_gain_bucketize", "capital_gain", "capital_gain_bucketize", TransformOp.BUCKETIZE, tf.float32, CAPITAL_GAIN_BOUNDARIES)
capital_loss_bucketize = FeatureTransformInfo("capital_loss_bucketize", "capital_loss", "capital_loss_bucketize", TransformOp.BUCKETIZE, tf.float32, CAPITAL_LOSS_BOUNDARIES)
hours_per_week_bucketize = FeatureTransformInfo("hours_per_week_bucketize", "hours_per_week", "capital_gain_bucketize", TransformOp.BUCKETIZE, tf.float32, HOURS_BOUNDARIES)

group1 = FeatureTransformInfo("group1", ["workclass_lookup", "hours_per_week_bucketize", "capital_gain_bucketize", "capital_loss_bucketize"], "group1", TransformOp.GROUP, None, None)
group2 = FeatureTransformInfo("group2", ["education_hash", "marital_status_lookup", "relationship_lookup", "occupation_hash"], "group2", TransformOp.GROUP, None, None)
group3 = FeatureTransformInfo("group3", ["age_bucketize", "sex_lookup", "race_lookup", "native_country_hash"], "group3", TransformOp.GROUP, None, None)

group1_embedding_wide = FeatureTransformInfo("group1_embedding_wide", "group1", "group1_embedding_wide", TransformOp.EMBEDDING, tf.int32, 1)
group2_embedding_wide = FeatureTransformInfo("group2_embedding_wide", "group2", "group2_embedding_wide", TransformOp.EMBEDDING, tf.int32, 1)

group1_embedding_deep = FeatureTransformInfo("group1_embedding_deep", "group1", "group1_embedding_deep", TransformOp.EMBEDDING, tf.int32, 8)
group2_embedding_deep = FeatureTransformInfo("group2_embedding_deep", "group2", "group2_embedding_deep", TransformOp.EMBEDDING, tf.int32, 8)
group3_embedding_deep = FeatureTransformInfo("group3_embedding_deep", "group3", "group3_embedding_deep", TransformOp.EMBEDDING, tf.int32, 8)

