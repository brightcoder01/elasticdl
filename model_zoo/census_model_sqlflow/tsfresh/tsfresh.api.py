import pandas as pd 
from pandas import DataFrame
from tsfresh import extract_features

def build_roll_data(input_df, time_column, value_column, window):
    roll_data_column = None
    return roll_data_column

def add_ts_extract_features(
    input_df,
    id_column,
    time_column,
    value_columns,
    windows,
    settings=None):
    for window in windows:
        for value_column in value_columns:
            rolled_data = build_roll_data(input_df, id_column, time_column, value_column, window)
            extracted_features = extract_features(rolled_data, column_id=id_column, column_sort=time_column, column_value=value_column)

    output_df = None
    return output_df