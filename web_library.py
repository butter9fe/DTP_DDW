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
    if (ancestry is None):
        return 1
    
    ancestry_row = or_df.loc[or_df['ANC'] == ancestry.capitalize()]
    return float(ancestry_row["OR"].iloc[0])

def hex_to_rgba(color: str, alpha: float) -> str:
    
    named_colors = {
        "red": "#FF0000",
        "orange": "#FFA500",
        "yellow": "#FFFF00",
        "lightgreen": "#90EE90",
        "green": "#008000",
        "gray": "#808080"
    }

    #convert to hex
    if color in named_colors:
        color = named_colors[color]

    #remove #
    color = color.lstrip('#')
    if len(color) != 6:
        raise ValueError(f"Invalid hex color format: {color}")
    
    #convert to RGBA
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"