from pathlib import Path
import pandas as pd

DATA_FILEPATH = Path('test_data_engineer.csv')
DATA_FILE_ENCODING = 'ISO-8859-8'


def load_data(filepath: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=filepath, encoding=encoding)
