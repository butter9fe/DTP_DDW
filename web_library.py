from constants import FILE_NAME
import pandas as pd

df: pd.DataFrame = pd.read_csv(FILE_NAME)

def get_country_variable(country: str, variable: str = "LCR"):
    country_row = df.loc[df['Country'] == country]
    return float(country_row[variable])

def get_variable_median(variable: str):
    row = df[variable]
    return row.median()