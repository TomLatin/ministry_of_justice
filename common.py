import logging
from pathlib import Path
import pandas as pd

DATA_FILEPATH = Path('test_data_engineer.csv')
DATA_FILE_ENCODING = 'ISO-8859-8'
DATA_KEY_COLUMN = "tnufa_endreasonName"
MAPPING_FILEPATH = Path("test_data_engineer_aggregated_case_result.csv")
MAPPING_FILE_ENCODING = 'ISO-8859-8'
MAPPING_KEY_COLUMN = 'tnufa_endreasonName'
MAPPING_VALUE_COLUMN = 'result_type'


def load_data(filepath: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=filepath, encoding=encoding)


def get_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)
