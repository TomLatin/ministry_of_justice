from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

from common import DATA_FILEPATH, DATA_FILE_ENCODING


def load_data(filepath: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=filepath, encoding=encoding)


def data_status(data: pd.DataFrame) -> None:
    print(f"data info:")
    print(f"{data.info()}\n")
    print(f"data describe:")
    print(f"{data.describe()}\n")
    print(f"data isnull:")
    print(f"{data.isnull().sum()}\n")
    print(f"data duplicated:")
    print(f"{data.duplicated().sum()}\n")
    # # Check how many rows have all None/NaN values
    num_all_none_rows = data.isnull().all(axis=1).sum()
    print("Number of rows where all values are None/NaN:", num_all_none_rows)


def build_profile(data: pd.DataFrame, output_file: str) -> None:
    profile = ProfileReport(data)
    profile.to_file(output_file)


if __name__ == '__main__':
    df = load_data(filepath=DATA_FILEPATH, encoding=DATA_FILE_ENCODING)
    data_status(data=df)
    build_profile(data=df, output_file='profiling_report.html')
