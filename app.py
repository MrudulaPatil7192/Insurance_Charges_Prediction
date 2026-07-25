import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Power BI inspired layout
st.set_page_config(
    page_title="Insurance Cost Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS targeting a Power BI Dark Dashboard aesthetic
st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    
    /* Power BI Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-left: 5px solid #F59E0B; /* Power BI Gold Accent */
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .metric-title {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #10B981; /* Green Accent */
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .card-container {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Custom Button Styling */
    .stButton>button {
        background-color: #F59E0B;
        color: #0F172A;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #D97706;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
MODEL_PATH = "decision_Regression_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found. Please ensure it is in the root directory.")
        return None

model = load_model()

# Header Area
st.title("📊 Insurance Premium Prediction Analytics")
st.markdown("<p style='color: #94A3B8;'>Power BI-Inspired Decision Tree Regression Dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar - User Input Panel
st.sidebar.header("⚙️ Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 30)
    sex = st.sidebar.selectbox("Sex", ["female", "male"])
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.sidebar.slider("Children / Dependents", 0, 10, 0)
    smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
    region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    # Map categorical features to numerical values if required by model preprocessing
    # Adjust mappings below if your model trained on direct categorical strings vs numbers
    sex_num = 1 if sex == "male" else 0
    smoker_num = 1 if smoker == "yes" else 0
    region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
    region_num = region_map[region]

    data = {
        "age": age,
        "sex": sex_num,      # Or use 'sex' if model expects string
        "bmi": bmi,
        "children": children,
        "smoker": smoker_num, # Or use 'smoker' if model expects string
        "region": region_num  # Or use 'region' if model expects string
    }
    return pd.DataFrame(data, index=[0]), {
        "Age": age, "Sex": sex.capitalize(), "BMI": bmi, 
        "Children": children, "Smoker": smoker.capitalize(), "Region": region.capitalize()
    }

input_df, display_dict = user_input_features()

# Main Dashboard Grid Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📋 Parameter Summary")
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    for key, value in display_dict.items():
        st.markdown(f"**{key}:** <span style='color: #F59E0B;'>{value}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Model Output & Analytics")
    
    if st.button("Calculate Estimated Premium"):
        if model is not None:
            try:
                # Perform Prediction
                prediction = model.predict(input_df)[0]
                
                # Display KPI Cards
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Predicted Annual Medical Insurance Cost</div>
                        <div class="metric-value">${prediction:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional Analytical Indicators
                m1, m2 = st.columns(2)
                with m1:
                    monthly_est = prediction / 12
                    st.markdown(f"""
                        <div class="card-container">
                            <div class="metric-title">Monthly Estimate</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: #38BDF8;">${monthly_est:,.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m2:
                    risk_category = "High Risk" if display_dict["Smoker"] == "Yes" or display_dict["BMI"] > 30 else "Standard Risk"
                    color = "#EF4444" if risk_category == "High Risk" else "#10B981"
                    st.markdown(f"""
                        <div class="card-container">
                            <div class="metric-title">Risk Assessment</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{risk_category}</div>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Model file could not be loaded properly.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B;'>AWS Deployed Decision Tree Regressor Dashboard</p>", unsafe_allow_html=True)
