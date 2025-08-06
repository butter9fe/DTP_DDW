import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import json
import urllib.request
from lcr_predictor import LungCancerPredictor
import constants
import web_library 

# Page configuration
st.set_page_config(
    page_title="LUCIA - Lung Cancer Insights & Action",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'user_risk' not in st.session_state:
    st.session_state.user_risk = 0.0

# Define variable mappings
VARIABLES = {
    "AST": "Surface Temperature",
    "HDI": "Human Development Index", 
    "OADR": "Old Age Dependency Ratio",
    "SR": "Smoking Rates",
    "OOP": "Out-of-pocket Health Expenditure",
    "ANC": "Ancestry"
}

# All features for comparison
ALL_FEATURES = {
    "EPI": "Environmental Performance Index",
    "AST": "Surface Temperature", 
    "AQI": "Air Quality Index",
    "HDI": "Human Development Index",
    "GDP": "GDP per Capita",
    "PWD_A": "Population Weighted Density",
    "OADR": "Old Age Dependency Ratio",
    "SR": "Smoking Rates",
    "OOP": "Out-of-pocket Health Expenditure",
    "PDPC": "Physicians Density per Capita",
    "CO2": "CO2 Emissions",
    "HUM": "Humidity"
}

# Rate level configuration
RISK_CONFIG = {
    'min_score': 1,
    'max_score': 48,
    'levels': [
        (0.8, "Very High Rate", "red"),
        (0.6, "High Rate", "orange"), 
        (0.4, "Moderate Rate", "yellow"),
        (0.2, "Low Rate", "lightgreen"),
        (0.0, "Very Low Rate", "green")
    ]
}

@st.cache_data
def load_data():
    """Load and cache the LCR data"""
    return pd.read_csv(constants.FILE_NAME)

@st.cache_data
def load_geojson():
    """Load and cache GeoJSON data"""
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    with urllib.request.urlopen(url) as response:
        return json.load(response)

@st.cache_data
def load_country_locations():
    """Load and process country location data"""
    df = pd.read_csv("Data/country-capital-lat-long-population.csv")
    df["Country"] = df["Country"].str.strip()
    
    # Filter for required countries
    df_filtered = df[df["Country"].isin(constants.countries_list)]
    
    # Create location dictionary
    return {
        row["Country"]: {
            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),
            "zoom": 6.0
        }
        for _, row in df_filtered.iterrows()
    }

def validate_inputs(inputs):
    """Validate user inputs and return validation results"""
    errors = []
    validated_inputs = {}
    
    for var_code, value in inputs.items():
        if var_code == "ANC":
            if not value or value.strip() == "":
                errors.append(f"{VARIABLES[var_code]} is required")
            else:
                validated_inputs[var_code] = value.strip()
        else:
            if not value or value.strip() == "":
                errors.append(f"{VARIABLES[var_code]} is required")
            else:
                try:
                    float_val = float(value.strip())
                    validated_inputs[var_code] = float_val
                except ValueError:
                    errors.append(f"{VARIABLES[var_code]} must be a valid number")
    
    return validated_inputs, errors

def predict_risk(inputs):
    """Predict risk using the model"""
    try:
        predictor = LungCancerPredictor(inputs)
        return predictor.predict_lcr()
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return 0.0

def get_risk_label_color(score):
    """Get risk label and color based on normalized score"""
    normalized = normalize_score(score)
    for threshold, label, color in RISK_CONFIG['levels']:
        if normalized >= threshold:
            return label, color
    return "Unknown", "gray"

def normalize_score(score):
    """Normalize score between 0 and 1"""
    min_score, max_score = RISK_CONFIG['min_score'], RISK_CONFIG['max_score']
    return max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))

def get_country_rank(lcr_data, user_risk):
    """Get user's rank and surrounding countries"""
    lcr_data_sorted = lcr_data.sort_values('LCR')
    
    # Find where user would rank
    user_rank = (lcr_data_sorted['LCR'] < user_risk).sum() + 1
    
    # Get surrounding countries (±2 positions)
    start_idx = max(0, user_rank - 3)
    end_idx = min(len(lcr_data_sorted), user_rank + 2)
    
    surrounding = lcr_data_sorted.iloc[start_idx:end_idx].copy()
    
    # Insert user data
    user_row = pd.DataFrame([{"Country": "Your Country", "LCR": user_risk}])
    
    # Combine and sort
    result = pd.concat([surrounding, user_row], ignore_index=True)
    result = result.sort_values('LCR').reset_index(drop=True)
    
    return result, user_rank

def create_ranking_chart(lcr_data, user_risk):
    """Create ranking comparison chart"""
    chart_data, user_rank = get_country_rank(lcr_data, user_risk)
    
    # Color mapping
    colors = ['lightblue' if country != "Your Country" else 'red' 
              for country in chart_data["Country"]]
    
    fig = go.Figure(data=[
        go.Bar(x=chart_data["Country"], 
               y=chart_data["LCR"],
               marker_color=colors,
               text=chart_data["LCR"].round(2),
               textposition='auto')
    ])
    
    fig.update_layout(
        title=f"Your Ranking: #{user_rank} out of {len(lcr_data)} countries",
        xaxis_title="Country",
        yaxis_title="Lung Cancer Rate (LCR)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_detailed_comparison(lcr_data, user_inputs, selected_country):
    """Create detailed comparison chart between user and selected country"""
    country_data = lcr_data[lcr_data['Country'] == selected_country].iloc[0]
    
    # Prepare comparison data
    comparison_vars = []
    user_vals = []
    country_vals = []
    
    # Add LCR comparison
    comparison_vars.extend(["LCR (You)", "LCR (Country)"])
    user_vals.extend([st.session_state.user_risk, country_data['LCR']])
    country_vals.extend([st.session_state.user_risk, country_data['LCR']])
    
    # Add other variables
    for var_code in ['AST', 'HDI', 'OADR', 'SR', 'OOP']:
        if var_code in user_inputs:
            comparison_vars.extend([f"{var_code} (You)", f"{var_code} (Country)"])
            user_vals.extend([user_inputs[var_code], country_data[var_code]])
            country_vals.extend([user_inputs[var_code], country_data[var_code]])
    
    # Create alternating data for grouped bars
    categories = []
    values = []
    sources = []
    
    for i in range(0, len(comparison_vars), 2):
        var_name = comparison_vars[i].split(' (')[0]
        categories.extend([var_name, var_name])
        values.extend([user_vals[i], user_vals[i+1]])
        sources.extend(['You', selected_country])
    
    df_comparison = pd.DataFrame({
        'Variable': categories,
        'Value': values,
        'Source': sources
    })
    
    fig = px.bar(df_comparison, 
                 x='Variable', 
                 y='Value', 
                 color='Source',
                 barmode='group',
                 title=f"Detailed Comparison: You vs {selected_country}",
                 color_discrete_map={'You': 'red', selected_country: 'lightblue'})
    
    fig.update_layout(height=500)
    return fig

def assess_variable_quality(user_inputs, lcr_data):
    """Assess how user's variables compare to dataset medians"""
    assessments = {}
    
    for var_code, value in user_inputs.items():
        if var_code != "ANC" and var_code in lcr_data.columns:
            median = lcr_data[var_code].median()
            
            # Determine if higher is better or worse based on variable type
            if var_code in ['HDI']:  # Higher is better
                if value < median * 0.8:
                    assessments[var_code] = ("Too low", "red")
                elif value > median * 1.2:
                    assessments[var_code] = ("Excellent", "green")
                else:
                    assessments[var_code] = ("Good", "lightgreen")
            
            elif var_code in ['SR', 'OOP', 'AST', 'OADR']:  # Lower is better
                if value > median * 1.2:
                    assessments[var_code] = ("Too high", "red")
                elif value < median * 0.8:
                    assessments[var_code] = ("Excellent", "green")
                else:
                    assessments[var_code] = ("Good", "lightgreen")
    
    return assessments

def get_policy_recommendations(assessments):
    """Get policy recommendations based on assessments"""
    recommendations = []
    
    for var_code, (status, color) in assessments.items():
        if status in ["Too high", "Too low"]:
            key = (var_code, status)
            if key in constants.policy_recommendations:
                recommendations.extend(constants.policy_recommendations[key][:2])  # Limit to 2 per variable
    
    return recommendations[:5]  # Limit total recommendations

def process_geojson_data(geojson, lcr_data):
    """Process GeoJSON data with risk scores"""
    min_score, max_score = RISK_CONFIG['min_score'], RISK_CONFIG['max_score']
    
    for feature in geojson["features"]:
        country = feature["properties"]["name"]
        country_data = lcr_data[lcr_data["Country"] == country]
        
        if not country_data.empty:
            raw_score = float(country_data.iloc[0]["LCR"])
            normalized = normalize_score(raw_score)
            
            feature["properties"]["risk_score"] = normalized
            feature["properties"]["raw_score"] = raw_score
            risk_label, _ = get_risk_label_color(raw_score)
            feature["properties"]["risk_level"] = risk_label
        else:
            feature["properties"]["risk_score"] = None
            feature["properties"]["raw_score"] = None
            feature["properties"]["risk_level"] = "No Data"

def main():
    """Main application logic"""
    # Load data
    lcr_data = load_data()
    geojson_data = load_geojson()
    country_locations = load_country_locations()
    
    # Process GeoJSON with risk data
    process_geojson_data(geojson_data, lcr_data)
    
    # Sidebar for input form
    with st.sidebar:
        st.title('🫁 LUCIA - Lung Cancer Insights & Action')
        st.markdown("Enter your country's indicators:")
        
        with st.form(key="indicator_form"):
            form_inputs = {}
            
            # Numeric inputs
            for var_code, var_label in VARIABLES.items():
                if var_code != "ANC":
                    current_value = st.session_state.inputs.get(var_code, "")
                    form_inputs[var_code] = st.text_input(
                        f"{var_label}",
                        value=str(current_value) if current_value else "",
                        key=f"input_{var_code}",
                        help=f"Enter {var_label.lower()}"
                    )
            
            # Ancestry dropdown
            form_inputs["ANC"] = st.selectbox(
                "Ancestry",
                options=["", "European", "Asian", "African"],
                index=0 if not st.session_state.inputs.get("ANC") else 
                      ["", "European", "Asian", "African"].index(st.session_state.inputs.get("ANC")),
                key="input_ANC"
            )
            
            submitted = st.form_submit_button("Calculate Rate", type="primary")
            
            if submitted:
                validated_inputs, errors = validate_inputs(form_inputs)
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.session_state.inputs = validated_inputs
                    st.session_state.user_risk = predict_risk(validated_inputs)
                    st.session_state.submitted = True
                    st.success("✅ Rate calculated successfully!")
                    st.rerun()
    
    # Main content area - 2 columns
    col1, col2 = st.columns([5, 2], gap='medium')
    
    # Column 1: Map
    with col1:
        st.header('🌍 Global Rate Map')
        
        # Country selection dropdown
        countries = ["Global View"] + sorted(list(country_locations.keys()))
        selected_country = st.selectbox("Focus on country:", countries, key="country_selector")
        
        # Set view state
        if selected_country != "Global View" and selected_country in country_locations:
            view_data = country_locations[selected_country]
        else:
            view_data = {"lat": 20, "lon": 0, "zoom": 1.5}
        
        view_state = pdk.ViewState(
            latitude=view_data["lat"],
            longitude=view_data["lon"],
            zoom=view_data["zoom"]
        )
        
        # Create map layer
        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson_data,
            get_fill_color="properties.risk_score != None ? [properties.risk_score * 255, (1 - properties.risk_score) * 255, 100, 180] : [0, 0, 0, 0]",
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255, 100],
            line_width_min_pixels=1
        )
        
        # Display map
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{name}</b><br/>Rate Level: {risk_level}<br/>LCR Score: {raw_score}",
                    "style": {"color": "white"}
                }
            ),
            key=f"map_{selected_country.lower().replace(' ', '_')}"
        )
    
    # Column 2: Rate Assessment Results
    if st.session_state.submitted:
        # Display risk score
        risk_score = st.session_state.user_risk
        risk_label, risk_color = get_risk_label_color(risk_score)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid {risk_color}; border-radius: 10px; background-color: rgba(255,255,255,0.1)">
            <h2>Predicted Rate Score</h2>
            <h1 style="color: {risk_color}; font-size: 3em">{risk_score:.2f}</h1>
            <h3 style="color: {risk_color}">{risk_label}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Variable Assessment")
        
        # Assess variables
        assessments = assess_variable_quality(st.session_state.inputs, lcr_data)
        
        for var_code, (status, color) in assessments.items():
            value = st.session_state.inputs[var_code]
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 5px; margin: 5px 0; background-color: {color}; border-radius: 5px; color: white if {color} == 'red' else black">
                <span><strong>{VARIABLES.get(var_code, var_code)}:</strong> {value}</span>
                <span>{status}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Policy recommendations
        recommendations = get_policy_recommendations(assessments)
        if recommendations:
            st.markdown("### 📋 Policy Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
    
    else:
        st.info("Complete the form in the sidebar to see your risk assessment.")
    
    # Column 3: Comparison Charts
    with col2:
        st.header('📊 Comparison')
        
        if st.session_state.submitted:
            if selected_country == "Global View":
                # Show ranking chart
                st.subheader("Global Ranking")
                fig = create_ranking_chart(lcr_data, st.session_state.user_risk)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Show detailed comparison with selected country
                if selected_country in lcr_data['Country'].values:
                    st.subheader(f"vs {selected_country}")
                    fig = create_detailed_comparison(lcr_data, st.session_state.inputs, selected_country)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {selected_country}")
        else:
            st.info("Calculate your risk to see comparisons.")

if __name__ == "__main__":
    main()