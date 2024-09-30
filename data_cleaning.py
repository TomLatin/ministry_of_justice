from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from common import DATA_FILEPATH, DATA_FILE_ENCODING, load_data

# Constants
MAPPING_FILEPATH = Path("test_data_engineer_aggregated_case_result.csv")
MAPPING_FILE_ENCODING = 'ISO-8859-8'
MAPPING_KEY_COLUMN = 'tnufa_endreasonName'
MAPPING_VALUE_COLUMN = 'result_type'
DATA_KEY_COLUMN = "tnufa_endreasonName"


def process_data(data_filepath: Path, data_file_encoding: str, data_key_column: str, mapping_filepath: Path,
                 mapping_file_encoding: str, mapping_key_column: str, mapping_value_column: str) -> pd.DataFrame:
    """
    Main function to process the data:
    - Load and clean the data
    - Replace values using the mapping file
    - Detect and convert date columns
    - Normalize float and date columns
    - One-hot encode categorical columns
    """
    df = load_data(data_filepath, data_file_encoding)
    df = df.drop_duplicates()
    df = replace_values_using_mapping(df, data_key_column, mapping_filepath, mapping_file_encoding, mapping_key_column,
                                      mapping_value_column)
    # Detect and convert date columns
    df, date_columns = detect_and_convert_dates(df)
    # Identify columns
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in date_columns]
    float_cols = df.select_dtypes(include=['float64']).columns.tolist()

    # Normalize float and date columns
    df = normalize_columns(df, float_cols, date_columns)
    # One-hot encode categorical columns
    df = one_hot_encode(df, categorical_cols)
    return df


def replace_values_using_mapping(df: pd.DataFrame, key_col: str, mapping_filepath: Path, mapping_file_encoding: str,
                                 mapping_key_column: str, mapping_value_column: str) -> pd.DataFrame:
    key_value_df = load_data(filepath=mapping_filepath, encoding=mapping_file_encoding)
    key_value_dict = dict(zip(key_value_df[mapping_key_column], key_value_df[mapping_value_column]))
    df.loc[:, key_col] = df[key_col].replace(key_value_dict)
    return df


def detect_and_convert_dates(df: pd.DataFrame) -> (pd.DataFrame, list):
    date_columns = []
    valid_year_range = (1900, 2100)  # Adjust this range based on your requirements

    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Convert to datetime
            converted = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

            # Check if all converted values are NaT (meaning no valid dates)
            if converted.notna().any():
                # Filter out NaT values and check for valid year range
                valid_dates = converted[(converted.notna()) & (converted.dt.year >= valid_year_range[0]) & (
                            converted.dt.year <= valid_year_range[1])]

                if not valid_dates.empty:
                    df[col] = converted
                    date_columns.append(col)
        except (ValueError, TypeError):
            pass

    return df, date_columns


def normalize_columns(df: pd.DataFrame, float_cols: list, date_cols: list) -> pd.DataFrame:
    scaler = MinMaxScaler()

    # Replace NaN values with -1 for the specified columns
    df[float_cols] = df[float_cols].fillna(-1)

    # Convert date columns to Unix timestamps (seconds since epoch)
    df[date_cols] = df[date_cols].apply(lambda col: col.astype('int64') // 10 ** 9)

    # Normalize the columns
    df[float_cols + date_cols] = scaler.fit_transform(df[float_cols + date_cols])
    return df


def one_hot_encode(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    # encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_col_names = encoder.get_feature_names_out(categorical_cols)

    # Convert to DataFrame and assign meaningful column names
    encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df.index)

    # Drop original categorical columns and concatenate the encoded columns
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, encoded_df], axis=1)

    return df


if __name__ == '__main__':
    processed_data = process_data(DATA_FILEPATH, DATA_FILE_ENCODING, DATA_KEY_COLUMN, MAPPING_FILEPATH,
                                  MAPPING_FILE_ENCODING, MAPPING_KEY_COLUMN, MAPPING_VALUE_COLUMN)
