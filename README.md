# Guide to LUCIA UI 
LUCIA - Lung Cancer Insights & Action


LUCIA is a Streamlit-powered web application designed to predict and visualize the national risk level of lung cancer using environmental, health, and socioeconomic indicators. The tool offers real-time personalized risk assessments and policy recommendations based on a custom predictive model.

---

##  Key Features

- **Personalized Prediction** using multiple health and environmental indicators
-  **Interactive Global Map** showing country-level risk distributions
-  **Dynamic Risk Comparison** via cross-country comparison
-  **Variable Quality Assessment** to identify areas of concern/improvement
-  **Policy Recommendation Engine** based on user data deviations
-  **Cluster & Ranking Visualization** with bar and charts

---

## 🏁 Getting access to UI

#0. streamlit run DDW.py in terminal

#1. Input respective country details: Surface Temp, Human Development Index, Old Age Dependency Ratio, Smoking Ratio, Out-of-pocket expenditure Ratio, and Ancestry group in side bar. (If not applicable to certain variables, do leave them blank)

#2. Press Calculate Rate

#3. Please check the respective attention score below the map with the risk level indicated.

#4. Sufficiency level of each variable is indicated for insight, with suggestions of targeted policy levers

#5. Select a country in Map Overview to view the selected country's risk level & LCR score and assess cross-country comparison of risk level under Global Ranking.

##  How our model works

-   Users input values such as smoking rate, surface temperature, HDI, etc.
    
-   The model uses the `LungCancerPredictor` class to calculate a risk score (LCR)
    
-   Risk score is normalized and categorized from “Minimal” to “Extreme”
    
-   The system visually compares user risk with global countries and gives targeted advice


## Variable metadata

`SR`

Smoking Rate (%)

`HDI`

Human Development Index

`AST`

Surface Temperature (°C)

`OADR`

Old Age Dependency Ratio (%)

`OOP`

Out-of-pocket Health Expenditure (%)

`ANC`

Genetic Ancestry (Asian, European, African)

---

This project uses a number of powerful Python libraries for data processing, visualization, and interactive web app development. Below is a list of all key dependencies used in the project, along with brief descriptions of their roles.



##  Core dependencies of library import

### `streamlit`
- Used for building the web app.
- Handles layout, input forms, session states, and rendering HTML/CSS.

### `pandas`
- Data loading, preprocessing, filtering, and transformation.

### `numpy` 
- Underpins operations in ML predictions and normalization.

---

## Visualization Libraries

### `plotly.express` & `plotly.graph_objects`
- For interactive bar charts and grouped comparisons between countries.
- Enables hoverable, zoomable, and dynamic plots.
- Chose to use this specific library as multiple graphs in streamlit or pandas as they could not be sorted and aligned accordingly.![
- ](https://imgur.com/a/3G2YKVR)

### `pydeck`
-  Used to render the global choropleth map with LCR scores.

---

## Geographic & Data Sources

### `urllib.request`
- Fetches GeoJSON files from online GitHub repository

### `json`
- processes GeoJSON data used in the world map.

---

## Project Specific Modules

### `constants.py`
-  Stores global constants like file paths, variable names, help texts, country list, and risk configuration.

### `lcr_predictor.py`
- Contains the `LungCancerPredictor` class, which applies your custom model to generate LCR predictions.

### `web_library.py`
- Includes helper functions for:
  - Median calculations
  - Ancestry odds ratios
  - Hex-to-RGBA color conversion
  - Policy recommendation mappings

---

##  Dataset Files

- `Data/country-capital-lat-long-population.csv` — Country geolocation data.
- `Data/LungCancer_Dataset.csv` — Main dataset with LCR and indicators.
- `Data/OddsRatio_Data.csv` — Genetic ancestry risk multipliers.


## Files
### 1. `DDW.py`

Main application script for the Streamlit web interface. It:

-   Collects user inputs via the sidebar
    
-   Loads and processes geospatial and health data
    
-   Predicts the lung cancer rate using user-provided inputs
    
-   Visualizes results including interactive maps, charts, and assessments
    

### 3. `constants.py`

Holds global constants such as:

-   Data file paths
    
-   List of countries to consider
    
-   Help texts for variables
    
-   Policy recommendations dictionary used throughout the app
    

### 4. `lcr_predictor.py`

Core predictive module:

-   Defines the `LungCancerPredictor` class
    
-   Loads and prepares data
    
-   Applies transformations and prediction model to compute Lung Cancer Rate (LCR)
    

### 5. `library.py`

Utility functions:

-   Normalization
    
-   Matrix transformations
    
-   Data formatting and manipulation
    

### 6. `model_builder.py`

Handles model construction:

-   Functions to build and train a regression model
    
-   Used by `lcr_predictor.py` to fit or load a model
    

### 7. `model_validator.py`

Performs model validation:

-   Computes prediction accuracy
    
-   Outputs metrics such as RMSE or R-squared
    

### 8. `web_library.py`

Web-specific utilities:

-   Functions to get medians, country-specific data
    
-   Converts hex to RGBA for consistent HTML rendering

### 8. `README.md`
- Current File

