import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="Heart Disease Risk Analytics 2026",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar - Theme Selection
st.sidebar.title("🫀 Menu & Pengaturan")
st.sidebar.markdown("---")

theme_choice = st.sidebar.radio(
    "🎨 Pilih Tema Tampilan:",
    ["🌙 Dark Mode", "☀️ Light Mode"],
    index=0
)

is_dark = "Dark" in theme_choice
plotly_template = "plotly_dark" if is_dark else "plotly_white"

# Custom CSS for Dark/Light Mode Styling
if is_dark:
    bg_color = "#0e1117"
    card_bg = "#1e222d"
    text_color = "#e0e6ed"
    subtext_color = "#94a3b8"
    border_color = "#2d3748"
    metric_bg = "#262c3a"
    accent_color = "#38bdf8"
    color_safe = "#4ade80"
    color_risk = "#f87171"
else:
    bg_color = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#1e293b"
    subtext_color = "#64748b"
    border_color = "#e2e8f0"
    metric_bg = "#f1f5f9"
    accent_color = "#0284c7"
    color_safe = "#16a34a"
    color_risk = "#dc2626"

custom_css = f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .metric-card {{
        background-color: {metric_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: {accent_color};
        margin-top: 4px;
    }}
    .metric-label {{
        font-size: 13px;
        font-weight: 600;
        color: {subtext_color};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .card-box {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }}
    div[data-testid="stSidebar"] {{
        background-color: {card_bg};
        border-right: 1px solid {border_color};
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
@st.cache_data
def load_dataset():
    paths = [
        "Prediction/heart_disease_risk_2026.csv",
        "heart_disease_risk_2026.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "patient_id" in df.columns:
                df.drop(columns=["patient_id"], inplace=True)
            return df
    st.error("❌ Dataset `heart_disease_risk_2026.csv` tidak ditemukan!")
    st.stop()

df_raw = load_dataset()

# Model Directory Resolver
def get_model_path(filename):
    paths = [
        os.path.join("Prediction", "model", filename),
        os.path.join("model", filename)
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ==============================================================================
# 3. SIDEBAR NAVIGATION & FILTERS
# ==============================================================================
st.sidebar.markdown("### 📍 Navigasi Halaman")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "📊 Dashboard Overview",
        "🧪 Analisis Klinis & Gaya Hidup",
        "🎯 Prediksi Risiko Pasien Baru",
        "📋 Eksplorasi Data"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filter Data Global")

# Global Filter Options
min_age = int(df_raw['age'].min())
max_age = int(df_raw['age'].max())
selected_age = st.sidebar.slider("Usia Pasien:", min_age, max_age, (min_age, max_age))

sex_options = ["Semua"] + list(df_raw['sex'].unique())
selected_sex = st.sidebar.selectbox("Jenis Kelamin:", sex_options)

smoker_options = ["Semua"] + list(df_raw['smoker_status'].unique())
selected_smoker = st.sidebar.selectbox("Status Merokok:", smoker_options)

target_options = ["Semua", "Tidak Sakit (0)", "Sakit Jantung (1)"]
selected_target = st.sidebar.selectbox("Status Penyakit Jantung:", target_options)

# Filter Dataframe
df = df_raw.copy()
df = df[(df['age'] >= selected_age[0]) & (df['age'] <= selected_age[1])]
if selected_sex != "Semua":
    df = df[df['sex'] == selected_sex]
if selected_smoker != "Semua":
    df = df[df['smoker_status'] == selected_smoker]
if selected_target == "Tidak Sakit (0)":
    df = df[df['has_heart_disease'] == 0]
elif selected_target == "Sakit Jantung (1)":
    df = df[df['has_heart_disease'] == 1]

st.sidebar.caption(f"Menampilkan **{len(df):,}** dari **{len(df_raw):,}** sampel data.")


# ==============================================================================
# 4. PAGE 1: DASHBOARD OVERVIEW
# ==============================================================================
if page == "📊 Dashboard Overview":
    st.title("🫀 Heart Disease Risk Dashboard 2026")
    st.markdown("Analisis komprehensif faktor risiko dan prevalensi penyakit jantung.")
    st.markdown("---")

    # KPI Metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_pts = len(df)
    sick_pts = df['has_heart_disease'].sum()
    prev_rate = (sick_pts / total_pts * 100) if total_pts > 0 else 0
    avg_age = df['age'].mean() if total_pts > 0 else 0
    avg_bp = df['resting_bp_systolic'].mean() if total_pts > 0 else 0
    avg_chol = df['cholesterol_total'].mean() if total_pts > 0 else 0

    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Pasien</div><div class='metric-value'>{total_pts:,}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Kasus Sakit</div><div class='metric-value' style='color:{color_risk}'>{sick_pts:,}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Prevalensi Risk</div><div class='metric-value' style='color:{color_risk}'>{prev_rate:.1f}%</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Rata-rata Usia</div><div class='metric-value'>{avg_age:.1f} thn</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Rata Systolic BP</div><div class='metric-value'>{avg_bp:.0f} mmHg</div></div>", unsafe_allow_html=True)
    with c6:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Rata Kolesterol</div><div class='metric-value'>{avg_chol:.0f} mg/dL</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1 Charts: Target Distribution & Gender/Age Breakdown
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("🍩 Distribusi Diagnosis Penyakit Jantung")
        target_counts = df['has_heart_disease'].value_counts().reset_index()
        target_counts.columns = ['Status', 'Jumlah']
        target_counts['Status_Label'] = target_counts['Status'].map({0: 'Aman (0)', 1: 'Sakit Jantung (1)'})
        
        fig_pie = px.pie(
            target_counts,
            values='Jumlah',
            names='Status_Label',
            color='Status_Label',
            color_discrete_map={'Aman (0)': color_safe, 'Sakit Jantung (1)': color_risk},
            hole=0.55,
            template=plotly_template
        )
        fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.05])
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("👥 Prevalensi Berdasarkan Kelompok Usia & Gender")
        df_age = df.copy()
        df_age['Kelompok Usia'] = pd.cut(df_age['age'], bins=[17, 35, 50, 65, 100], labels=['18-35', '36-50', '51-65', '65+'])
        age_sex_grouped = df_age.groupby(['Kelompok Usia', 'sex', 'has_heart_disease']).size().reset_index(name='Jumlah')
        age_sex_grouped['Status'] = age_sex_grouped['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})

        fig_bar = px.bar(
            age_sex_grouped,
            x='Kelompok Usia',
            y='Jumlah',
            color='Status',
            facet_col='sex',
            barmode='group',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            template=plotly_template
        )
        fig_bar.update_layout(margin=dict(t=30, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 Charts: Chest Pain & Smoker Status Impact
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("🫁 Jenis Nyeri Dada (Chest Pain) vs Risiko")
        cp_df = df.groupby(['chest_pain_type', 'has_heart_disease']).size().reset_index(name='Jumlah')
        cp_df['Status'] = cp_df['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})
        
        fig_cp = px.bar(
            cp_df,
            x='chest_pain_type',
            y='Jumlah',
            color='Status',
            barmode='group',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            template=plotly_template,
            labels={'chest_pain_type': 'Tipe Nyeri Dada'}
        )
        fig_cp.update_layout(height=360, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cp, use_container_width=True)

    with col4:
        st.subheader("🚬 Status Merokok vs Risiko Penyakit Jantung")
        smk_df = df.groupby(['smoker_status', 'has_heart_disease']).size().reset_index(name='Jumlah')
        smk_df['Status'] = smk_df['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})
        
        fig_smk = px.bar(
            smk_df,
            x='smoker_status',
            y='Jumlah',
            color='Status',
            barmode='group',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            template=plotly_template,
            labels={'smoker_status': 'Status Merokok'}
        )
        fig_smk.update_layout(height=360, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_smk, use_container_width=True)

    # Row 3: Correlation Matrix
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔥 Heatmap Korelasi Antar Fitur Numerik")
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr().round(2)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r" if is_dark else "Blues",
        template=plotly_template
    )
    fig_corr.update_layout(height=650, margin=dict(t=30, b=20, l=20, r=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# 5. PAGE 2: ANALISIS KLINIS & GAYA HIDUP
# ==============================================================================
elif page == "🧪 Analisis Klinis & Gaya Hidup":
    st.title("🧪 Analisis Mendalam Faktor Klinis & Gaya Hidup")
    st.markdown("Eksplorasi hubungan fitur medis, indikator laboratorium, serta kebiasaan sehari-hari.")
    st.markdown("---")

    # Section 1: Tekanan Darah & Lipid Profile
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🩸 Tekanan Darah Systolic vs Diastolic")
        df_bp = df.copy()
        df_bp['Status'] = df_bp['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})
        fig_bp = px.scatter(
            df_bp,
            x='resting_bp_systolic',
            y='resting_bp_diastolic',
            color='Status',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            hover_data=['age', 'bmi', 'chest_pain_type'],
            labels={
                'resting_bp_systolic': 'Systolic BP (mmHg)',
                'resting_bp_diastolic': 'Diastolic BP (mmHg)',
                'Status': 'Status Diagnosis'
            },
            template=plotly_template,
            render_mode='svg',
            opacity=0.7
        )
        fig_bp.update_layout(height=400)
        st.plotly_chart(fig_bp, use_container_width=True)

    with col2:
        st.subheader("🥑 Distribusi Kolesterol (Total, HDL, LDL, Triglycerides)")
        lipid_cols = ['cholesterol_total', 'hdl', 'ldl', 'triglycerides']
        df_lipid = df.melt(id_vars=['has_heart_disease'], value_vars=lipid_cols, var_name='Lipid', value_name='mg/dL')
        df_lipid['Status'] = df_lipid['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})

        fig_lipid = px.box(
            df_lipid,
            x='Lipid',
            y='mg/dL',
            color='Status',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            template=plotly_template
        )
        fig_lipid.update_layout(height=400)
        st.plotly_chart(fig_lipid, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Sugar & HbA1c, Stress & Daily Steps
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🍬 Gula Darah Puasa vs HbA1c (%)")
        df_sugar = df.copy()
        df_sugar['Status'] = df_sugar['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})
        fig_sugar = px.scatter(
            df_sugar,
            x='fasting_blood_sugar',
            y='hba1c',
            color='Status',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            labels={'fasting_blood_sugar': 'Gula Darah Puasa (mg/dL)', 'hba1c': 'HbA1c (%)', 'Status': 'Status Diagnosis'},
            template=plotly_template,
            render_mode='svg',
            opacity=0.7
        )
        fig_sugar.update_layout(height=400)
        st.plotly_chart(fig_sugar, use_container_width=True)

    with col4:
        st.subheader("🏃 Langkah Harian (Daily Steps) vs Stress Score")
        df_stress = df.copy()
        df_stress['Status'] = df_stress['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})
        fig_stress = px.scatter(
            df_stress,
            x='daily_steps',
            y='stress_score',
            color='Status',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            labels={'daily_steps': 'Jumlah Langkah Harian', 'stress_score': 'Tingkat Stress (0-100)', 'Status': 'Status Diagnosis'},
            template=plotly_template,
            render_mode='svg',
            opacity=0.7
        )
        fig_stress.update_layout(height=400)
        st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: Multivariate Scatter (SVG compatible)
    st.subheader("🌐 Visualisasi Multivariat: Usia vs Kolesterol vs Max Heart Rate")
    viz_mode = st.radio("Pilih Mode Visualisasi:", ["📊 2D Scatter / Bubble Chart (Rekomendasi - Tanpa WebGL)", "🌐 3D Scatter Plot (Butuh WebGL Browser)"], horizontal=True)

    df_multi = df.copy()
    df_multi['Status'] = df_multi['has_heart_disease'].map({0: 'Aman', 1: 'Sakit Jantung'})

    if "2D" in viz_mode:
        fig_multi = px.scatter(
            df_multi,
            x='age',
            y='cholesterol_total',
            size='max_heart_rate_achieved',
            color='Status',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            labels={
                'age': 'Usia (Tahun)',
                'cholesterol_total': 'Total Kolesterol (mg/dL)',
                'max_heart_rate_achieved': 'Max HR (Ukuran Bubble)',
                'Status': 'Status Diagnosis'
            },
            template=plotly_template,
            render_mode='svg',
            opacity=0.7
        )
        fig_multi.update_layout(height=500)
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        fig_3d = px.scatter_3d(
            df_multi,
            x='age',
            y='cholesterol_total',
            z='max_heart_rate_achieved',
            color='Status',
            color_discrete_map={'Aman': color_safe, 'Sakit Jantung': color_risk},
            labels={
                'age': 'Usia',
                'cholesterol_total': 'Total Kolesterol',
                'max_heart_rate_achieved': 'Max Heart Rate',
                'Status': 'Status Diagnosis'
            },
            template=plotly_template,
            opacity=0.8
        )
        fig_3d.update_layout(height=600, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_3d, use_container_width=True)


# ==============================================================================
# 6. PAGE 3: PREDIKSI RISIKO PASIEN BARU (INFERENCE)
# ==============================================================================
elif page == "🎯 Prediksi Risiko Pasien Baru":
    st.title("🎯 Prediksi Risiko Real-Time Pasien Baru")
    st.markdown("Masukkan data parameter kesehatan pasien di bawah ini untuk memprediksi tingkat risiko penyakit jantung menggunakan model Machine Learning yang telah dilatih.")
    st.markdown("---")

    # Load Model Artifacts
    rf_path = get_model_path("model_random_forest.pkl")
    xgb_path = get_model_path("model_xgboost.pkl")
    lr_path = get_model_path("model_logistic_regression.pkl")
    scaler_path = get_model_path("scaler.pkl")
    cols_path = get_model_path("feature_columns.pkl")

    models_loaded = {}
    scaler = None
    feature_columns = None

    if rf_path and xgb_path and lr_path and scaler_path and cols_path:
        try:
            models_loaded['Random Forest'] = joblib.load(rf_path)
            models_loaded['XGBoost'] = joblib.load(xgb_path)
            models_loaded['Logistic Regression'] = joblib.load(lr_path)
            scaler = joblib.load(scaler_path)
            feature_columns = joblib.load(cols_path)
            st.success("✅ Seluruh model (Random Forest, XGBoost, Logistic Regression) berhasil dimuat.")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat file model: {e}")
    else:
        st.warning("⚠️ File model (.pkl) tidak ditemukan di folder `model/`. Fitur prediksi menggunakan dummy/rule-based simulasi.")

    # Form Input Pasien Baru
    st.markdown("### 📝 Form Parameter Pasien")

    with st.form("patient_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("#### 👤 Demografi & Vital Sign")
            in_age = st.number_input("Usia (Tahun)", min_value=18, max_value=100, value=55)
            in_sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            in_bp_sys = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=140)
            in_bp_dia = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=130, value=90)
            in_bmi = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=55.0, value=28.5, step=0.1)
            in_hr_rest = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=130, value=78)
            in_hr_max = st.number_input("Max Heart Rate (bpm)", min_value=80, max_value=220, value=145)

        with col_b:
            st.markdown("#### 🩸 Profil Laboratorium & Klinis")
            in_chol = st.number_input("Total Kolesterol (mg/dL)", min_value=100, max_value=450, value=230)
            in_hdl = st.number_input("HDL (mg/dL)", min_value=15, max_value=120, value=45)
            in_ldl = st.number_input("LDL (mg/dL)", min_value=30, max_value=300, value=150)
            in_trig = st.number_input("Triglycerides (mg/dL)", min_value=30, max_value=500, value=180)
            in_fbs = st.number_input("Gula Darah Puasa (mg/dL)", min_value=50, max_value=300, value=115)
            in_hba1c = st.number_input("HbA1c (%)", min_value=3.5, max_value=14.0, value=6.2, step=0.1)
            in_st_dep = st.number_input("ST Depression", min_value=0.0, max_value=7.0, value=1.5, step=0.1)

        with col_c:
            st.markdown("#### 🩺 Gejala & Gaya Hidup")
            in_cp = st.selectbox("Tipe Nyeri Dada", ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"])
            in_angina = st.selectbox("Exercise Induced Angina", [True, False])
            in_fam_hist = st.selectbox("Riwayat Keluarga Sakit Jantung", [True, False])
            in_smoker = st.selectbox("Status Merokok", ["Never", "Former", "Current"])
            in_alcohol = st.number_input("Alkohol (Unit/Minggu)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            in_exercise = st.number_input("Olahraga (Menit/Minggu)", min_value=0, max_value=600, value=60)
            in_sleep = st.number_input("Tidur (Jam/Hari)", min_value=3.0, max_value=12.0, value=6.0, step=0.5)
            in_stress = st.number_input("Stress Score (0-100)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
            in_wearable = st.selectbox("Memiliki Device Wearable", [False, True])
            in_steps = st.number_input("Langkah Harian (Daily Steps)", min_value=500, max_value=25000, value=5000)
            in_diet = st.number_input("Skor Kualitas Diet (0-100)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)

        submit_btn = st.form_submit_button("🚀 Jalankan Prediksi Risiko", use_container_width=True)

    if submit_btn:
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis Risiko")

        # Build raw dict
        raw_patient = {
            'age': in_age,
            'sex': in_sex,
            'resting_bp_systolic': in_bp_sys,
            'resting_bp_diastolic': in_bp_dia,
            'cholesterol_total': in_chol,
            'hdl': in_hdl,
            'ldl': in_ldl,
            'triglycerides': in_trig,
            'fasting_blood_sugar': in_fbs,
            'hba1c': in_hba1c,
            'bmi': in_bmi,
            'resting_heart_rate': in_hr_rest,
            'max_heart_rate_achieved': in_hr_max,
            'chest_pain_type': in_cp,
            'exercise_induced_angina': int(in_angina),
            'st_depression': in_st_dep,
            'family_history': int(in_fam_hist),
            'smoker_status': in_smoker,
            'alcohol_units_per_week': in_alcohol,
            'exercise_minutes_per_week': in_exercise,
            'sleep_hours': in_sleep,
            'stress_score': in_stress,
            'wearable_owner': int(in_wearable),
            'daily_steps': in_steps,
            'diet_quality_score': in_diet
        }

        preds = {}
        probs = {}

        if models_loaded and feature_columns and scaler:
            # Preprocessing input to match training schema
            df_in = pd.DataFrame([raw_patient])
            df_in['sex'] = df_in['sex'].map({'Male': 1, 'Female': 0})
            df_in = pd.get_dummies(df_in, columns=['chest_pain_type', 'smoker_status'], drop_first=True)

            for col in feature_columns:
                if col not in df_in.columns:
                    df_in[col] = 0

            df_in = df_in[feature_columns]
            df_scaled = scaler.transform(df_in)

            for name, model_obj in models_loaded.items():
                X_input = df_scaled if name == 'Logistic Regression' else df_in
                p_val = model_obj.predict(X_input)[0]
                prob_val = model_obj.predict_proba(X_input)[0][1]
                preds[name] = int(p_val)
                probs[name] = float(prob_val)
        else:
            # Fallback heuristic simulation if models not loaded
            score = 0
            if in_age > 50: score += 2
            if in_bp_sys > 130: score += 2
            if in_chol > 200: score += 2
            if in_smoker != "Never": score += 1
            if in_cp == "Asymptomatic": score += 2
            prob_sim = min(0.98, max(0.05, score / 9.0))
            pred_sim = 1 if prob_sim >= 0.5 else 0
            
            for m in ['Random Forest', 'XGBoost', 'Logistic Regression']:
                preds[m] = pred_sim
                probs[m] = prob_sim

        avg_probability = np.mean(list(probs.values()))
        majority_vote = 1 if list(preds.values()).count(1) >= 2 else 0

        # Display Metrics
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("### 📋 Prediksi Per Model")
            for name in preds:
                p_text = "⚠️ Berisiko Sakit Jantung" if preds[name] == 1 else "✅ Aman / Risiko Rendah"
                p_color = color_risk if preds[name] == 1 else color_safe
                st.markdown(f"**{name}**: <span style='color:{p_color}; font-weight:bold;'>{p_text}</span> (Probabilitas: **{probs[name]*100:.1f}%**)", unsafe_allow_html=True)
            
            st.markdown("---")
            final_status = "⚠️ BERISIKO SAKIT JANTUNG" if majority_vote == 1 else "✅ AMAN / RISIKO RENDAH"
            final_color = color_risk if majority_vote == 1 else color_safe
            st.markdown(f"#### 🗳️ Consensus Voting (3 Model):")
            st.markdown(f"<h3 style='color:{final_color};'>{final_status}</h3>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("### 🧭 Gauge Meter Rata-rata Probabilitas")
            
            # Plotly Semi-Circle Gauge with Needle Arrow Pointer
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_probability * 100,
                number={'suffix': '%', 'font': {'size': 34, 'color': final_color}},
                title={'text': "Rata-rata Probabilitas Risiko", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_color},
                    'bar': {'color': 'rgba(0,0,0,0)', 'thickness': 0}, # Hide default bar so needle stands out clearly
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': border_color,
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(74, 222, 128, 0.3)'},
                        {'range': [50, 75], 'color': 'rgba(250, 204, 21, 0.3)'},
                        {'range': [75, 100], 'color': 'rgba(248, 113, 113, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': color_risk, 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            # Calculate needle arrow coordinates (0% = left 180°, 50% = top 90°, 100% = right 0°)
            theta = (1.0 - avg_probability) * np.pi
            cx, cy = 0.5, 0.22
            r = 0.31
            w = 0.015

            tx = cx + r * np.cos(theta)
            ty = cy + r * np.sin(theta)

            lx = cx + w * np.cos(theta + np.pi/2)
            ly = cy + w * np.sin(theta + np.pi/2)

            rx = cx + w * np.cos(theta - np.pi/2)
            ry = cy + w * np.sin(theta - np.pi/2)

            path_needle = f"M {lx} {ly} L {tx} {ty} L {rx} {ry} Z"

            # Add Sharp Needle Arrow Shape
            fig_gauge.add_shape(
                type="path",
                path=path_needle,
                xref="paper", yref="paper",
                fillcolor=final_color,
                line=dict(color=final_color, width=1.5)
            )

            # Add Pivot Hub Circle Shape at center
            fig_gauge.add_shape(
                type="circle",
                xref="paper", yref="paper",
                x0=cx-0.025, y0=cy-0.025,
                x1=cx+0.025, y1=cy+0.025,
                fillcolor="#1e293b" if is_dark else "#ffffff",
                line=dict(color=final_color, width=3)
            )

            fig_gauge.update_layout(
                height=320,
                margin=dict(t=30, b=10, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': text_color}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption("💡 Disclaimer: Prediksi ini berbasis model statistik ML dan bertujuan untuk pertimbangan pendukung, bukan pengganti diagnosis medis profesional.")


# ==============================================================================
# 7. PAGE 4: EKSPLORASI DATA
# ==============================================================================
elif page == "📋 Eksplorasi Data":
    st.title("📋 Eksplorasi & Download Dataset")
    st.markdown("Lihat tabel data mentah yang difilter, statistik deskriptif, serta opsi untuk mengunduh dataset.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Data Table", "📈 Statistik Deskriptif"])

    with tab1:
        st.dataframe(df, use_container_width=True, height=450)
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data CSV (Terfilter)",
            data=csv_data,
            file_name="heart_disease_risk_filtered.csv",
            mime="text/csv"
        )

    with tab2:
        st.markdown("### Statistik Deskriptif (Numerik)")
        st.dataframe(df.describe().T, use_container_width=True)
