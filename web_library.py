from constants import FILE_NAME
import pandas as pd

df: pd.DataFrame = pd.read_csv(FILE_NAME)

def get_country_variable(country: str, variable: str = "LCR"):
    country_row = df.loc[df['Country'] == country]
    return country_row[variable]

def get_variable_mean(variable: str):
    row = df[variable]
    return row.mean()

print(get_country_variable("Myanmar"))
print(get_country_variable("Myanmar", "AQI"))
print(get_variable_mean("GDP"))