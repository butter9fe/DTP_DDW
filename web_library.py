from constants import FILE_NAME, OR_FILE_NAME
import pandas as pd

df: pd.DataFrame = pd.read_csv(FILE_NAME)
or_df: pd.DataFrame = pd.read_csv(OR_FILE_NAME)

def get_country_variable(country: str, variable: str = "LCR"):
    country_row = df.loc[df['Country'] == country]
    return float(country_row[variable].iloc[0])

def get_variable_median(variable: str):
    row = df[variable]
    return row.median()

def get_or_from_ancestry(ancestry: str):
    ancestry_row = or_df.loc[or_df['ANC'] == ancestry.capitalize()]
    return float(ancestry_row["OR"].iloc[0])