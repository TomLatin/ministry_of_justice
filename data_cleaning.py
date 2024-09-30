from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from common import DATA_FILEPATH, DATA_FILE_ENCODING, load_data

MAPPING_FILEPATH = Path("test_data_engineer_aggregated_case_result.csv")
MAPPING_KEY_COLUMN = 'tnufa_endreasonName'
MAPPING_VALUE_COLUMN = 'result_type'
DATA_KEY_COLUMN = "tnufa_endreasonName"
def detect_date_columns(df):
    # Detect object columns that are actually dates and convert them to datetime
    date_columns = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col])
            date_columns.append(col)
        except (ValueError, TypeError):
            pass
    return df, date_columns

if __name__ == '__main__':
    df = load_data(filepath=DATA_FILEPATH, encoding=DATA_FILE_ENCODING)

    # Remove duplicates
    data_no_dup = df.drop_duplicates()

    # Replace values in column DATA_KEY_COLUMN with values from MAPPING_VALUE_COLUMN
    key_value_df = load_data(filepath=MAPPING_FILEPATH, encoding=DATA_FILE_ENCODING)
    key_value_dict = dict(zip(key_value_df[MAPPING_KEY_COLUMN], key_value_df[MAPPING_VALUE_COLUMN]))
    data_no_dup.loc[:, DATA_KEY_COLUMN] = data_no_dup[DATA_KEY_COLUMN].replace(key_value_dict)

    # One-hot encode
    # Step 1: Detect date columns and convert them to datetime
    data_no_dup, date_columns = detect_date_columns(data_no_dup)

    # Step 2: Identify categorical (object) columns that are not dates
    object_cols = data_no_dup.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in object_cols if col not in date_columns]

    # Step 3: Identify float columns
    float_cols = data_no_dup.select_dtypes(include=['float64']).columns.tolist()

    # Step 4: Normalize float and date columns using MinMaxScaler
    scaler = MinMaxScaler()

    # For dates, convert to numeric (e.g., timestamp) before normalization
    data_no_dup[date_columns] = data_no_dup[date_columns].apply(
        lambda col: col.astype('int64') // 10 ** 9)  # Convert to Unix timestamp

    # Apply MinMaxScaler to float and date columns
    data_no_dup[float_cols + date_columns] = scaler.fit_transform(data_no_dup[float_cols + date_columns])

    # Step 5: Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity
    encoded_cols = encoder.fit_transform(data_no_dup[categorical_cols])

    # Convert encoded columns back to DataFrame with proper column names
    # Get more meaningful column names: e.g., "columnname_categoryname"
    encoded_col_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=data_no_dup.index)

    # Step 6: Drop original categorical columns and concatenate encoded columns
    data_no_dup = data_no_dup.drop(categorical_cols, axis=1)
    data_no_dup = pd.concat([data_no_dup, encoded_df], axis=1)
    print(data_no_dup.head())
