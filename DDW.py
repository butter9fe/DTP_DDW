import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import json
import urllib.request
from lcr_predictor import LungCancerPredictor
import constants
import web_library 


st.set_page_config(
    page_title="AIRIS ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")


with st.sidebar:
    st.title('Fill in your details')
    

    def blanked_function(variable_name):
        st.warning(f"{variable_name} is missing. Please provide a value or check the input.")
    # You can add additional behavior here (e.g., default values, disabling prediction, etc.)

    # Define your variable names and display names
    variables = {
    "AST": "Surface Temperature",
    "HDI": "Human Development Index",
    "OADR": "Old Age Dependency Ratio",
    "SR": "Smoking Rates",
    "OOP": "Out-of-pocket Health Expenditure",
    "ANC": "Ancestry"
    }
    lcr = pd.read_csv(constants.FILE_NAME)

    # Create input fields and check for blanks
    inputs = {}
    st.header("Enter Country Indicators")
    
    def blanked_function(variable_name):
        return 0.1

    with st.form(key="indicator_form"):
        inputs = {}
        for var_code, var_label in variables.items():
            user_input = st.text_input(f"{var_label} ({var_code})", key=var_code)
            if user_input.strip() == "":
                inputs[var_code] = blanked_function(var_code)

        

        submitted = st.form_submit_button("Submit")
    
if submitted:
    # 1. 모델 예측
    model = LungCancerPredictor(inputs)
    user_risk = float(model.predict_lcr())

    lcr["diff"] = lcr["LCR"] - user_risk
    higher = lcr[lcr["diff"] > 0].sort_values("diff").head(2)
    lower = lcr[lcr["diff"] < 0].sort_values("diff", key=abs).head(2)
    user_row = pd.DataFrame([{"Country": "Country 1", "LCR": user_risk}])
    chart_df = pd.concat([lower, user_row, higher])

col = st.columns(( 4.5, 2), gap='medium')
import csv

country_list = constants.countries_list
# ✅ CSV 파일 불러오기 (수도 기준 좌표 데이터)
df = pd.read_csv("country-capital-lat-long-population.csv")

# ✅ 국가 이름 정리
df["Country"] = df["Country"].str.strip()

# ✅ 필터링: 필요한 국가만
df_filtered = df[df["Country"].isin(country_list)]

# ✅ Zoom 조건 정의
small_countries = {"Singapore", "Luxembourg", "Belgium", "Malta"}

# ✅ 딕셔너리 생성
country_locations = {
    row["Country"]: {
        "lat": float(row["Latitude"]),
        "lon": float(row["Longitude"]),
        "zoom": float(row["Zoom"])
    }
    for _, row in df_filtered.iterrows()
}

with col[1]:
    st.header("Comparison")

    countries = list(country_locations.keys())
    selected_country = st.selectbox("Select a country to zoom", [""] + countries)

    if submitted:
        with col[1]:
            user_row = pd.DataFrame([{"Country": "Your country", "LCR": user_risk}])
            lcr["diff"] = lcr["LCR"] - user_risk
            higher = lcr[lcr["diff"] > 0].sort_values("diff").head(2)
            lower = lcr[lcr["diff"] < 0].sort_values("diff", key=abs).head(2)

            chart_df = pd.concat([lower, user_row, higher])
            chart_df = chart_df.reset_index(drop=True)

            # y축 범위 조정
            buffer = 0.05
            y_min = max(0, chart_df["LCR"].min() - buffer)
            y_max = min(1, chart_df["LCR"].max() + buffer)

            fig = px.bar(
                chart_df,
                x="Country",
                y="LCR",
                color="LCR",
                color_continuous_scale="RdYlGn_r",
                title="Risk Factor Comparison"
            )

            fig.update_layout(
                yaxis=dict(rangemode="tozero"),
                title_font_size=20,
                yaxis_title="Predicted Risk Factor",
                xaxis_title=None
            )

            st.plotly_chart(fig, use_container_width=True)
    

    

# ======= pydeck map 출력 (왼쪽 col[0]) =======


with col[0]:
    st.header('AIRIS: National Respiratory Risk Calculator')
    st.markdown("Please enter your country indicators on the left sidebar to get started.")
    st.write("Predicted Values")

    @st.cache_data
    def load_geojson():
        url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        with urllib.request.urlopen(url) as response:
            return json.load(response)
    geojson = load_geojson()
        
    if selected_country and selected_country in country_locations:
        view_data = country_locations[selected_country]
    else:
        view_data = {"lat": 10, "lon": 20, "zoom": 1.5}  # default global view
        
    # view 설정 (드롭다운 선택 값에 따라 결정)
    view_state = pdk.ViewState(
        latitude=view_data["lat"],
        longitude=view_data["lon"],
        zoom=view_data["zoom"]
    )

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        get_fill_color="[properties.risk_score * 255, (1 - properties.risk_score) * 255, 100, 180]",
        pickable=True,
        auto_highlight=True,
    )
    min_score = 3
    max_score = 40
    def get_risk_label(score):
        if score >= 0.8:
            return "Very Dangerous"
        elif score >= 0.7:
            return "Moderately Dangerous"
        elif score >= 0.5:
            return "Moderate"
        elif score >= 0.3:
            return "Safe"
        else:
            return "Very Safe"

    for feature in geojson["features"]:
        country = feature["properties"]["name"]
        row = lcr[lcr["Country"] == country]

        if not row.empty:
            raw_score = float(row.iloc[0]["LCR"])
            normalized = (raw_score - min_score) / (max_score - min_score)
            normalized = max(0.0, min(1.0, normalized))
            feature["properties"]["risk_score"] = normalized
            feature["properties"]["risk_level"] = get_risk_label(normalized)
        else:
            feature["properties"]["risk_score"] = None
            feature["properties"]["risk_level"] = "No information"
            
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
            "html": "<b>{name}</b><br/>Risk Level: {risk_level}",
            "style": {
                "color": "white"
            }
        }
        ),
        key=selected_country.lower() if selected_country else "global"
    )
    

    
    if submitted:
        try:
            # None 아닌 값만 추출 (예: 예측 모델 요구에 맞게 조정 가능)
            valid_inputs = {k: v for k, v in inputs.items() if v is not None}

            # 모든 변수가 필요할 경우: 아래 예처럼 key 직접 지정
            
            st.metric("Predicted Risk Score", round(user_risk, 3))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

        

#conda activate streamlit_env

#streamlit run DDW.py
