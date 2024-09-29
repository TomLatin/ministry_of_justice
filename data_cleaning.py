import pandas as pd

data = pd.read_csv(filepath_or_buffer='test_data_engineer.csv', encoding='ISO-8859-8')
data = data.drop_duplicates()