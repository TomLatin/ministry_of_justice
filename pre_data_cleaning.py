import pandas as pd
from ydata_profiling import ProfileReport

from common import DATA_FILEPATH, DATA_FILE_ENCODING, load_data, get_logging


def data_status(data: pd.DataFrame) -> None:
    logger.info(msg=f"data info:")
    logger.info(msg=f"{data.info()}\n")
    logger.info(msg=f"data describe:")
    logger.info(msg=f"{data.describe()}\n")
    logger.info(msg=f"data isnull:")
    logger.info(msg=f"{data.isnull().sum()}\n")
    logger.info(msg=f"data duplicated:")
    logger.info(msg=f"{data.duplicated().sum()}\n")
    # Check how many rows have all None/NaN values
    num_all_none_rows = data.isnull().all(axis=1).sum()
    logger.info(msg=f"Number of rows where all values are None/NaN:{num_all_none_rows}")


def build_profile(data: pd.DataFrame, output_file: str) -> None:
    profile = ProfileReport(data)
    profile.to_file(output_file)


if __name__ == '__main__':
    logger = get_logging()
    df = load_data(filepath=DATA_FILEPATH, encoding=DATA_FILE_ENCODING)
    data_status(data=df)
    build_profile(data=df, output_file='profiling_report.html')
