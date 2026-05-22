import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Accident Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

# =========================
# THEME / STYLING
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #F7F9FC;
    }

    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
    }

    .kpi-title {
        font-size: 16px;
        color: #555;
    }

    .kpi-value {
        font-size: 34px;
        font-weight: bold;
        color: #1F3C88;
    }

    h1, h2, h3 {
        color: #1F3C88;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD DATA
# =========================
@st.cache_data

def load_data():
    df = pd.read_csv("accidents_clean.csv")

    if 'accident_date' in df.columns:
        df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')
        df['year'] = df['accident_date'].dt.year
        df['month'] = df['accident_date'].dt.month_name()

    return df


df = load_data()

st.title("🚗 Accident Analytics & Prediction Dashboard")
st.markdown("MSc Business Intelligence & Analytics Dissertation")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Dashboard Filters")

if 'accident_region' in df.columns:
    regions = st.sidebar.multiselect(
        "Select Region",
        options=df['accident_region'].dropna().unique(),
        default=df['accident_region'].dropna().unique()
    )
    df = df[df['accident_region'].isin(regions)]

if 'vehicle_type' in df.columns:
    vehicles = st.sidebar.multiselect(
        "Vehicle Type",
        options=df['vehicle_type'].dropna().unique(),
        default=df['vehicle_type'].dropna().unique()
    )
    df = df[df['vehicle_type'].isin(vehicles)]

if 'year' in df.columns:
    years = st.sidebar.multiselect(
        "Select Year",
        options=sorted(df['year'].dropna().unique()),
        default=sorted(df['year'].dropna().unique())
    )
    df = df[df['year'].isin(years)]

# =========================
# DASHBOARD 1
# EXECUTIVE OVERVIEW
# =========================
st.header("📊 Dashboard 1 — Executive Overview")

col1, col2, col3, col4 = st.columns(4)

# KPIs

total_accidents = len(df)

fatal_accidents = 0
if 'is_fatal' in df.columns:
    fatal_accidents = df['is_fatal'].sum()

fatality_rate = 0
if total_accidents > 0:
    fatality_rate = (fatal_accidents / total_accidents) * 100

night_accidents = 0
if 'is_night' in df.columns:
    night_accidents = df['is_night'].sum()

with col1:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Accidents</div>
            <div class='kpi-value'>{total_accidents:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Fatal Accidents</div>
            <div class='kpi-value'>{fatal_accidents:,}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Fatality Rate</div>
            <div class='kpi-value'>{fatality_rate:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Night Accidents</div>
            <div class='kpi-value'>{night_accidents:,}</div>
        </div>
    """, unsafe_allow_html=True)

left, right = st.columns(2)

# Monthly Trends
with left:
    if 'month' in df.columns:
        monthly = df.groupby('month').size().reset_index(name='accidents')

        fig = px.line(
            monthly,
            x='month',
            y='accidents',
            title='Monthly Accident Trends',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

# Fatal vs Non Fatal
with right:
    if 'is_fatal' in df.columns:
        fatal_df = df['is_fatal'].value_counts().reset_index()
        fatal_df.columns = ['Type', 'Count']

        fig = px.pie(
            fatal_df,
            names='Type',
            values='Count',
            title='Fatal vs Non-Fatal Accidents',
            hole=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

# Top Dangerous Regions
if 'accident_region' in df.columns:
    dangerous = df.groupby('accident_region').size().reset_index(name='count')
    dangerous = dangerous.sort_values(by='count', ascending=False).head(10)

    fig = px.bar(
        dangerous,
        x='accident_region',
        y='count',
        title='Top Dangerous Regions'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# DASHBOARD 2
# GEOGRAPHIC ANALYSIS
# =========================
st.header("🗺️ Dashboard 2 — Geographic Risk Analysis")

geo_col1, geo_col2 = st.columns(2)

with geo_col1:
    if 'accident_region' in df.columns:
        region_counts = df.groupby('accident_region').size().reset_index(name='count')

        fig = px.treemap(
            region_counts,
            path=['accident_region'],
            values='count',
            title='Accident Distribution by Region'
        )

        st.plotly_chart(fig, use_container_width=True)

with geo_col2:
    if 'accident_district' in df.columns:
        district_counts = df.groupby('accident_district').size().reset_index(name='count')
        district_counts = district_counts.sort_values(by='count', ascending=False).head(15)

        fig = px.bar(
            district_counts,
            x='count',
            y='accident_district',
            orientation='h',
            title='Top Dangerous Districts'
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# DASHBOARD 3
# DRIVER ANALYSIS
# =========================
st.header("👨‍✈️ Dashboard 3 — Driver Risk Analysis")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    if 'driver_age' in df.columns:
        fig = px.histogram(
            df,
            x='driver_age',
            nbins=20,
            title='Driver Age Distribution'
        )

        st.plotly_chart(fig, use_container_width=True)

with risk_col2:
    if 'driver_gender' in df.columns:
        gender_counts = df.groupby('driver_gender').size().reset_index(name='count')

        fig = px.bar(
            gender_counts,
            x='driver_gender',
            y='count',
            title='Accidents by Gender'
        )

        st.plotly_chart(fig, use_container_width=True)

if 'alcohol_involvement' in df.columns:
    alcohol_counts = df.groupby('alcohol_involvement').size().reset_index(name='count')

    fig = px.bar(
        alcohol_counts,
        x='alcohol_involvement',
        y='count',
        title='Alcohol Involvement in Accidents'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# DASHBOARD 4
# VEHICLE ANALYTICS
# =========================
st.header("🚘 Dashboard 4 — Vehicle Analytics")

veh_col1, veh_col2 = st.columns(2)

with veh_col1:
    if 'vehicle_brand' in df.columns:
        brands = df.groupby('vehicle_brand').size().reset_index(name='count')
        brands = brands.sort_values(by='count', ascending=False).head(10)

        fig = px.bar(
            brands,
            x='vehicle_brand',
            y='count',
            title='Top Vehicle Brands in Accidents'
        )

        st.plotly_chart(fig, use_container_width=True)

with veh_col2:
    if 'vehicle_age_clean' in df.columns:
        fig = px.histogram(
            df,
            x='vehicle_age_clean',
            nbins=15,
            title='Vehicle Age Distribution'
        )

        st.plotly_chart(fig, use_container_width=True)

if 'vehicle_type' in df.columns:
    vehicle_type = df.groupby('vehicle_type').size().reset_index(name='count')

    fig = px.pie(
        vehicle_type,
        names='vehicle_type',
        values='count',
        title='Vehicle Type Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# DASHBOARD 5
# MACHINE LEARNING
# =========================
st.header("🤖 Dashboard 5 — Predictive Analytics")

if 'is_fatal' in df.columns:

    ml_df = df.copy()

    # Select numeric columns
    numeric_cols = ml_df.select_dtypes(include=np.number).columns.tolist()

    if 'is_fatal' in numeric_cols:
        numeric_cols.remove('is_fatal')

    if len(numeric_cols) > 0:

        X = ml_df[numeric_cols].fillna(0)
        y = ml_df['is_fatal']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)

        ml_col1, ml_col2 = st.columns(2)

        with ml_col1:
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=accuracy * 100,
                title={'text': 'Model Accuracy'},
                gauge={'axis': {'range': [0, 100]}}
            ))

            st.plotly_chart(fig, use_container_width=True)

        with ml_col2:
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            })

            importances = importances.sort_values(by='importance', ascending=False).head(10)

            fig = px.bar(
                importances,
                x='importance',
                y='feature',
                orientation='h',
                title='Top Feature Importances'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)

        cm_df = pd.DataFrame(
            cm,
            index=['Actual Non-Fatal', 'Actual Fatal'],
            columns=['Predicted Non-Fatal', 'Predicted Fatal']
        )

        st.subheader('Confusion Matrix')
        st.dataframe(cm_df)

# =========================
# RAW DATA
# =========================
st.header("📄 Dataset Preview")
st.dataframe(df.head(100))

st.markdown("---")
st.markdown("Created for MSc Business Intelligence & Analytics Dissertation")
