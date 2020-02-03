from collections import namedtuple

FeatureInfo = namedtuple("FeatureInfo", ["name", "op_name", "dtype", "param"])
FeatureTransformInfo = namedtuple(
    "FeatureTransformInfo",
    ["name", "input_name", "output_name", "op_name", "output_dtype", "param"],
)
SchemaInfo = namedtuple("SchemaInfo", ["name", "dtype"])


class TransformOp(object):
    HASH = "HASH"
    BUCKETIZE = "BUCKETIZE"
    LOOKUP = "LOOKUP"
    EMBEDDING = "EMBEDDING"
    GROUP = "GROUP"
    ARRAY = "ARRAY"


def get_id_boundaries(feature_group):
    boundaries = [0]
    for feature_info in feature_group:
        boundaries.append(boundaries[-1] + get_max_id(feature_info))
    return boundaries


def get_max_id(feature_info):
    if feature_info.op_name == TransformOp.LOOKUP:
        return len(feature_info.param) + 1
    elif feature_info.op_name == TransformOp.HASH:
        return feature_info.param
    elif feature_info.op_name == TransformOp.BUCKETIZE:
        return len(feature_info.param) + 1
    else:
        raise ValueError("The op is not supported")
