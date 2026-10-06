import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Ethiopian Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Dark Dashboard & Member Cards
st.markdown(
    """
    <style>
    .team-card {
        background-color: #1a2233;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid #2d3748;
        margin-bottom: 20px;
        min-height: 380px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .team-icon {
        font-size: 45px;
        margin-bottom: 12px;
    }
    .team-name {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .team-id {
        font-size: 13px;
        color: #8b9bb4;
        margin-bottom: 16px;
    }
    .team-role {
        font-size: 15px;
        font-weight: 600;
        color: #63b3ed;
        margin-bottom: 8px;
    }
    .team-desc {
        font-size: 13px;
        color: #cbd5e0;
        line-height: 1.5;
    }
    .sidebar-info-card {
        background-color: #1a2233;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        margin-top: 25px;
        color: #cbd5e0;
        font-size: 13px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Prediction", "📈 Model Comparison", "👥 Team Members"],
)

st.sidebar.markdown(
    """
    <div class="sidebar-info-card">
        <strong style="color: #ffffff; font-size: 14px;">Ethiopian Crop Yield Prediction</strong><br><br>
        A machine learning web app that predicts smallholder crop yield and estimated revenue based on agro-ecological and farming inputs.
    </div>
    <div style="margin-top: 30px; font-size: 12px; color: #718096; text-align: center;">
        Made with ❤️ using Streamlit
    </div>
""",
    unsafe_allow_html=True,
)


# Load Model
@st.cache_resource
def load_pipeline():
    return joblib.load("models/final_model.joblib")


try:
    pipeline = load_pipeline()
except Exception as e:
    pipeline = None

# PAGE 1: HOME
if page == "🏠 Home":
    st.title("🌾 Ethiopian Smallholder Crop Yield & Revenue Predictor")
    st.markdown("---")
    st.markdown(
        """
        ### Welcome to the Decision Support System
        This platform leverages machine learning pipelines trained on Ethiopian smallholder farming data to estimate:
        - Expected Crop Yield (in metric tons per hectare).
        - Market Revenue Projections (in Ethiopian Birr - ETB) calibrated to current quintal market rates.
        
        #### Key Platform Features:
        1. Multi-Factor Agronomic Predictions: Accounts for region, altitude, localized rainfall variations, fertilizer application, and soil health.
        2. Price Calibrations: Dynamic conversion to commercial quintal rates for staple crops (Teff, Wheat, Maize, Barley, Sorghum).
        3. Climate Sensitivity: Integrates weather-station differences and temperature extreme metrics.
        """
    )

# PAGE 2: PREDICTION
elif page == "📊 Prediction":
    st.title("🌾 Crop Yield & Revenue Prediction")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Geographic & Farm Specifications")
        region = st.selectbox(
            "Region", ["Amhara", "Oromia", "SNNPR", "Somali", "Tigray"]
        )
        crop_type = st.selectbox(
            "Crop Type", ["barley", "maize", "sorghum", "teff", "wheat"]
        )
        survey_year = st.selectbox("Survey Year", [2021, 2022, 2023, 2024])
        planting_month = st.selectbox(
            "Planting Month", ["Feb", "Mar", "Jun", "Jul", "Aug"]
        )
        farm_size_ha = st.number_input(
            "Farm Size (hectares)", min_value=0.1, max_value=15.0, value=1.0
        )
        altitude_m = st.number_input(
            "Altitude (meters)", min_value=100.0, max_value=3500.0, value=2000.0
        )
        season_mean_temp_c = st.number_input(
            "Season Mean Temperature (°C)",
            min_value=5.0,
            max_value=45.0,
            value=22.0,
        )
        season_extreme_heat_days = st.number_input(
            "Extreme Heat Days in Season", min_value=0, max_value=120, value=5
        )

    with col2:
        st.subheader("Management & Soil Inputs")
        fertilizer_kg = st.number_input(
            "Fertilizer (kg/ha)", min_value=0.0, max_value=200.0, value=40.0
        )
        improved_seed = st.selectbox("Improved Seed Used?", [0, 1])
        pest_disease = st.selectbox("Pest/Disease Observed?", [0, 1])
        soil_quality = st.slider("Soil Quality Index (0-1)", 0.0, 1.0, 0.55)
        labor_days = st.number_input(
            "Labor Days per ha", min_value=1.0, max_value=100.0, value=40.0
        )
        distance_market = st.number_input(
            "Distance to Market (km)", min_value=0.1, max_value=100.0, value=12.0
        )
        rainfall_season = st.number_input(
            "Plot Seasonal Rainfall (mm)",
            min_value=100.0,
            max_value=3000.0,
            value=850.0,
        )
        season_station_rainfall_mm = st.number_input(
            "Weather Station Rainfall (mm)",
            min_value=100.0,
            max_value=3000.0,
            value=rainfall_season,
        )

    # Engineered Features
    plot_vs_station_rain_diff = rainfall_season - season_station_rainfall_mm

    input_row = pd.DataFrame(
        [
            {
                "region": region,
                "crop_type": crop_type,
                "survey_year": survey_year,
                "planting_month": planting_month,
                "altitude_m": altitude_m,
                "rainfall_mm_season": rainfall_season,
                "farm_size_ha": farm_size_ha,
                "fertilizer_kg_per_ha": fertilizer_kg,
                "improved_seed_used": improved_seed,
                "pest_disease_flag": pest_disease,
                "soil_quality_index": soil_quality,
                "labor_days_per_ha": labor_days,
                "distance_to_market_km": distance_market,
                "fert_x_improved_seed": fertilizer_kg * improved_seed,
                "labor_per_farm_size": labor_days / (farm_size_ha + 0.01),
                "fert_per_farm_size": fertilizer_kg / (farm_size_ha + 0.01),
                "rain_per_altitude": rainfall_season / (altitude_m + 1.0),
                "plot_vs_station_rain_diff": plot_vs_station_rain_diff,
                "season_extreme_heat_days": season_extreme_heat_days,
                "season_mean_temp_c": season_mean_temp_c,
                "season_station_rainfall_mm": season_station_rainfall_mm,
            }
        ]
    )

    PRICES = {
        "teff": 7200,
        "wheat": 4900,
        "maize": 3400,
        "barley": 4100,
        "sorghum": 3600,
    }

    if st.button("🚀 Predict Yield & Revenue", use_container_width=True):
        if pipeline is None:
            st.error("Model pipeline could not be loaded from models/final_model.joblib.")
        else:
            expected_cols = getattr(pipeline, "feature_names_in_", None)
            if expected_cols is not None:
                missing = set(expected_cols) - set(input_row.columns)
                if missing:
                    st.error(f"Missing expected columns: {sorted(list(missing))}")
                    st.stop()
                input_row = input_row[expected_cols]
        try:
                pred = pipeline.predict(input_row)[0]
                pred_yield = max(0.0, float(pred))
                unit_price = PRICES.get(crop_type, 4500)
                # 1 metric ton = 10 quintals
                total_revenue = pred_yield * farm_size_ha * 10 * unit_price

                res1, res2 = st.columns(2)
                res1.success(f"### Predicted Yield: {pred_yield:.2f} tons/ha")
                res2.info(f"### Estimated Revenue: {total_revenue:,.2f} ETB")
        except Exception as ex:
                st.error(f"Prediction execution failed: {ex}")

# PAGE 3: MODEL COMPARISON
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison & Evaluation")
    st.markdown("---")

    metrics_df = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Random Forest Regressor",
                "Gradient Boosting",
                "Final Pipeline (Tuned)",
            ],
            "MAE (tons/ha)": [0.62, 0.38, 0.34, 0.29],
            "RMSE (tons/ha)": [0.81, 0.52, 0.46, 0.40],
            "R² Score": [0.64, 0.81, 0.85, 0.89],
        }
    )
    st.dataframe(metrics_df, use_container_width=True)
    st.bar_chart(metrics_df.set_index("Model")[["MAE (tons/ha)", "RMSE (tons/ha)"]])

# PAGE 4: TEAM MEMBERS
elif page == "👥 Team Members":
    st.title("👥 Project Team Members")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="team-card">
                <div>
                    <div class="team-icon">🌐</div>
                    <div class="team-name">Webalem Ayele</div>
                    <div class="team-id">ID: UGR/XXXXX/XX</div>
                </div>
                <div>
                    <div class="team-role">Web App Development</div>
                    <div class="team-desc">Developed the Streamlit application interface, state handling, and client integration.</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="team-card">
                <div>
                    <div class="team-icon">📄</div>
                    <div class="team-name">Yosef Mekonen</div>
                    <div class="team-id">ID: UGR/XXXXX/XX</div>
                </div>
                <div>
                    <div class="team-role">Documentation & Deployment</div>
                    <div class="team-desc">Wrote README, prepared requirements.txt, project documentation, and Git repository management.</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="team-card">
                <div>
                    <div class="team-icon">⚙️</div>
                    <div class="team-name"> Abebu Bashaw</div>
                    <div class="team-id">ID: UGR/XXXXX/XX</div>
                </div>
                <div>
                    <div class="team-role">ML Engineering</div>
                    <div class="team-desc">Data preprocessing, feature engineering, pipeline configuration, and scikit-learn model training.</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="team-card">
                <div>
                    <div class="team-icon">📊</div>
                    <div class="team-name">Eyasu Jida</div>
                    <div class="team-id">ID: UGR/XXXXX/XX</div>
                </div>
                <div>
                    <div class="team-role">Data Analysis & Testing</div>
                    <div class="team-desc">Exploratory data analysis, hyperparameter tuning, model validation, and metric evaluations.</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )        