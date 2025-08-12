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
from textwrap import dedent
import plotly.io as pio

# Page configuration
st.set_page_config(
    page_title="LUCIA",
    page_icon="🔓",
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
        (0.8, "Extreme risk level", "red"),
        (0.6, "High risk level", "orange"), 
        (0.4, "Moderate risk level", "yellow"),
        (0.2, "Low risk level", "lightgreen"),
        (0.0, "Minimal risk level", "green")
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
    df = pd.read_csv("Data/location_zoom_country.csv")
    df["Country"] = df["Country"].str.strip()
    
    # Filter for required countries
    df_filtered = df[df["Country"].isin(constants.countries_list)]
    
    # Create location dictionary
    return {
        row["Country"]: {
            "lat": float(row["Latitude "]),
            "lon": float(row["Longitude "]),
            "zoom": float(row["Zoom"])
        }
        for _, row in df_filtered.iterrows()
    }

def validate_inputs(inputs):
    """Validate user inputs and return validation results"""
    errors = []
    validated_inputs = {}

    # Whole list is empty!
    print(inputs.values())
    if (all(not item for item in inputs.values())):
        errors.append("Please enter at least 1 value!")
    
    for var_code, value in inputs.items():
        if not value:
            validated_inputs[var_code] = None # To just use mean
        elif var_code == "ANC":
            validated_inputs[var_code] = value.strip()
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
HIGHER_BETTER = {'HDI'}
LOWER_BETTER = {'SR', 'OOP', 'AST', 'OADR'}

def compute_variable_severity(user_inputs: dict, lcr_data: pd.DataFrame) -> dict[str, float]:
    """
    Returns a non-negative severity score per variable.
    Score is 0 when the value deviates in a *good* direction,
    and >0 when it deviates in the *bad* direction.
    Scale: fraction away from median (e.g., 0.25 = 25% away in bad direction).
    """
    severities: dict[str, float] = {}

    for var_code, val in user_inputs.items():
        if var_code not in lcr_data.columns or val is None or var_code == "ANC":
            continue

        median = lcr_data[var_code].median()
        if median is None or pd.isna(median) or median == 0:
            # Avoid divide-by-zero; skip or treat as 0 severity
            severities[var_code] = 0.0
            continue

        ratio = (val - median) / median

        if var_code in LOWER_BETTER:
            # Bad only when value > median (higher than typical = worse)
            severity = max(0.0, ratio)  # e.g., 0.30 means 30% too high
        elif var_code in HIGHER_BETTER:
            # Bad only when value < median (lower than typical = worse)
            severity = max(0.0, -ratio) # e.g., 0.20 means 20% too low
        else:
            # Unknown direction: penalize absolute deviation (conservative)
            severity = abs(ratio)

        severities[var_code] = float(severity)

    return severities

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
        if var_code != "ANC" and var_code in lcr_data.columns and value:
            median = lcr_data[var_code].median()
            
            # Determine if higher is better or worse based on variable type
            if var_code in ['HDI']:  # Higher is better
                if value < median * 0.8:
                    assessments[var_code] = ("Insufficient", web_library.hex_to_rgba("red", 0.4))
                elif value > median * 1.2:
                    assessments[var_code] = ("Ideal", web_library.hex_to_rgba("green", 0.4))
                else:
                    assessments[var_code] = ("Acceptable", web_library.hex_to_rgba("lightgreen", 0.4))
            
            elif var_code in ['SR', 'OOP', 'AST', 'OADR']:  # Lower is better
                if value > median * 1.2:
                    assessments[var_code] = ("Excessive", web_library.hex_to_rgba("red", 0.4))
                elif value < median * 0.8:
                    assessments[var_code] = ("Ideal", web_library.hex_to_rgba("green", 0.4))
                else:
                    assessments[var_code] = ("Acceptable", web_library.hex_to_rgba("lightgreen", 0.4))
    
    return assessments

def get_policy_recommendations(assessments: dict, severities: dict, max_total: int = 5) -> list[str]:
    """
    Pick policies for variables with the highest 'bad-direction' severity first.
    Only variables not 'Acceptable' are considered. Ties follow dict order.
    """
    # Filter to non-acceptable variables and sort by severity desc
    ranked_vars = sorted(
        (v for v in assessments.keys() if assessments[v][0] != "Acceptable"),
        key=lambda v: severities.get(v, 0.0),
        reverse=True
    )

    recs: list[str] = []
    for var_code in ranked_vars:
        if var_code in constants.policy_recommendations:
            #take top 2 per variable 
            for r in constants.policy_recommendations[var_code][:2]:
                recs.append(r)
                if len(recs) >= max_total:
                    return recs
    return recs

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

def create_circular_progress_chart(score, risk_label, risk_color):
    """Create a circular progress chart for the attention score"""
    # Normalize score to percentage (0-100)
    normalized_score = normalize_score(score) * 100
    
    # Create the circular progress chart
    fig = go.Figure()
    
    # Create a donut chart with custom styling
    fig.add_trace(go.Pie(
        values=[normalized_score, 100 - normalized_score],
        hole=0.75,  # Larger hole for cleaner look
        showlegend=False,
        textinfo='none',
        hoverinfo='skip',
        marker=dict(
            colors=[risk_color, '#E5E7EB'],  # Progress color and light gray background
            line=dict(width=0)
        ),
        sort=False,
        direction='clockwise',
        rotation=90  # Start from top
    ))
    
    # Update layout for clean appearance
    fig.update_layout(
        width=180,
        height=180,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                x=0.5, y=0.5,  # Use relative positioning (0.5 = center)
                text=f"<b>{normalized_score:.0f}%</b>",
                showarrow=False,
                font=dict(size=24, color='#FFFFFF', family='Arial, sans-serif'),
                align='center',
                xref='paper',  # Reference to paper coordinates
                yref='paper'   # Reference to paper coordinates
            )
        ]
    )
    
    return fig

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
        st.markdown("Enter country details")
        st.caption("Leave this field blank if not applicable")
        
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
                        help=constants.HELP_TEXTS.get(var_code, f"Enter {var_label.lower()}")
                    )
            
            # Ancestry dropdown
            form_inputs["ANC"] = st.selectbox("Ancestry",
                options=["", "European", "Asian", "African"],
                index=0 if not st.session_state.inputs.get("ANC") else 
                      ["", "European", "Asian", "African"].index(st.session_state.inputs.get("ANC")),
                key="input_ANC",
                help=constants.HELP_TEXTS.get("ANC", "Select your ancestry")
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
                    st.success(" Rate calculated successfully!")
                    st.rerun()
    st.image('Data/banner_image.jpg')
    
    def create_variable_progress_chart(var_code, value, status, rgba_color, var_label):
        """Create a circular progress chart for individual variables"""
        # Define ranges for different variables
        ranges = {
            'AST': (0, 45),      # Surface Temperature
            'HDI': (0, 1),       # Human Development Index  
            'OADR': (0, 100),    # Old Age Dependency Ratio
            'SR': (0, 100),      # Smoking Rates
            'OOP': (0, 100)      # Out-of-pocket Health Expenditure
        }
        
        # Get the range for this variable
        min_val, max_val = ranges.get(var_code, (0, 100))
        
        # Calculate percentage based on the variable's range
        if value is not None:
            percentage = min(100, max(0, ((value - min_val) / (max_val - min_val)) * 100))
        else:
            percentage = 0
        
        # Extract RGB values from rgba_color for better control
        # Convert status to color
        color_map = {
            'Ideal': '#10B981',      # Green
            'Acceptable': '#84CC16',  # Light green  
            'Insufficient': '#EF4444', # Red
            'Excessive': '#EF4444'    # Red
        }
        progress_color = color_map.get(status, '#6B7280')
        
        # Create the circular progress chart
        fig = go.Figure()
        
        # Create a donut chart
        fig.add_trace(go.Pie(
            values=[percentage, 100 - percentage],
            hole=0.75,
            showlegend=False,
            textinfo='none',
            hoverinfo='skip',
            marker=dict(
                colors=[progress_color, '#E5E7EB'],
                line=dict(width=0)
            ),
            sort=False,
            direction='clockwise',
            rotation=90
        ))
        
        # Update layout
        fig.update_layout(
            width=120,
            height=120,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    x=0.5, y=0.5,
                    text=f"<b>{value:.1f}</b>" if value is not None else "<b>-</b>",
                    showarrow=False,
                    font=dict(size=14, color='#FFFFFF', family='Arial, sans-serif'),
                    align='center',
                    xref='paper',
                    yref='paper'
                )
            ]
        )
        
        return fig

    # Main content area - 2 columns
    col1, col2 = st.columns([5, 2], gap='medium')
    
    # Column 1: Map
    with col1:
        
        st.subheader('Map Overview')
        
        # Country selection dropdown
        countries = ["Global View"] + sorted(list(country_locations.keys()))
        selected_country = st.selectbox("Insight for:", countries, key="country_selector")
        
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

        
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        if st.session_state.submitted:
            # Display risk score
            risk_score = st.session_state.user_risk
            risk_label, risk_color = get_risk_label_color(risk_score)
            
            chart_col, content_col = st.columns([1, 3])
            
            with chart_col:
                # Create and display circular progress chart
                progress_fig = create_circular_progress_chart(risk_score, risk_label, risk_color)
                st.plotly_chart(progress_fig, use_container_width=True, config={'displayModeBar': False})
            
            with content_col:
                st.markdown("### Attention Score Analysis")
                st.markdown(f"**Current Score:** {risk_score:.2f} / 50")
                st.markdown(f"**Risk Level:** {risk_label}")
                st.markdown("This score represents predicted risk of lung cance provided environmental and demographic factors.")
                
                
            st.write("")
            st.write("")
            
            

            st.markdown("### Variable Assessment")
            st.write("")

            assessments = assess_variable_quality(st.session_state.inputs, lcr_data)
            severities  = compute_variable_severity(st.session_state.inputs, lcr_data)

            if not assessments:
                st.info("No variables to assess. Please enter at least one value in the sidebar.")
            else:
                codes = list(assessments.keys())
                cols = st.columns(len(codes))  # one row, left→right

                for i, var_code in enumerate(codes):
                    status, rgba_color = assessments[var_code]
                    value = st.session_state.inputs.get(var_code)
                    var_label = VARIABLES.get(var_code, var_code)

                    fig = create_variable_progress_chart(var_code, value, status, rgba_color, var_label)
                    with cols[i]:
                        # Keep the figure compact so they fit in one row
                        fig.update_layout(width=160, height=160, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=False,
                                        config={"displayModeBar": False},
                                        key=f"var_progress_{var_code}_{i}")

                        disp_val = f"{value:.2f}" if isinstance(value, (int, float)) else "-"
                        st.markdown(
                            f"""
                            <div style="text-align:center; line-height:1.2; margin-top:-8px;">
                                <div style="font-size:14px; font-weight:700;">{var_label}</div>
                                <div style="font-size:11px; color:#bbb;">Value: {disp_val}</div>
                                <div style="font-size:11px; color:#bbb;">Assessment: {status}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            st.write("")
            st.write("")

            # Policy recommendations
            recommendations = get_policy_recommendations(assessments, severities)
            if recommendations:
                st.markdown("### 🏳️ Policy Recommendations")

                # CSS for card-like boxes
                st.markdown(
                    """
                    <style>
                    .policy-card {
                        background-color: #1e1e1e;
                        padding: 12px 16px;
                        border-radius: 8px;
                        box-shadow: 1px 1px 5px rgba(0,0,0,0.3);
                        margin-bottom: 10px;
                        font-size: 14px;
                        line-height: 1.4;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Render each recommendation inside its own box
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"<div class='policy-card'><b>{i}.</b> {rec}</div>", unsafe_allow_html=True)
            else:
                st.write("#### Keep it up! 🎉")
        
        else:
            st.info("Complete the form in the sidebar to see your risk assessment.")
    
    
    
    # Column 2: Comparison Charts
    with col2:
        st.subheader('Cross-country comparison')
        
        if st.session_state.submitted:
            if selected_country == "Global View":
                # Show ranking chart
                st.subheader("Global Ranking")
                fig = create_ranking_chart(lcr_data, st.session_state.user_risk)
                st.plotly_chart(fig, use_container_width=True, key = 'ranking_chart_global')
                
            else:
                # Show detailed comparison with selected country
                if selected_country in lcr_data['Country'].values:
                    st.subheader(f"vs {selected_country}")
                    fig = create_detailed_comparison(lcr_data, st.session_state.inputs, selected_country)
                    safe_country_key = selected_country.lower().replace(" ", "_")
                    st.plotly_chart(fig, use_container_width=True, key=f"compare_chart_{safe_country_key}")
                else:
                    st.warning(f"No data available for {selected_country}")
        else:
            st.info("Calculate your risk to see comparisons.")

if __name__ == "__main__":
    main()