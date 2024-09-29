import pandas as pd
from ydata_profiling import ProfileReport

# Load the data
data = pd.read_csv(filepath_or_buffer='test_data_engineer.csv', encoding='ISO-8859-8')
print(f"data info:\n{data.info()}\n")
print(f"data describe:\n{data.describe()}\n")
print(f"data isnull:\n{data.isnull().sum()}\n")
print(f"data duplicated:\n{data.duplicated().sum()}\n")

profile = ProfileReport(data, title="Profiling Report")
profile.to_file("profiling_report.html")