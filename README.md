# 🫀 Heart Disease Risk Prediction 2026

> **Machine Learning Pipeline & Interactive Web Dashboard** untuk memprediksi risiko penyakit jantung menggunakan tiga algoritma klasifikasi (Logistic Regression, Random Forest, XGBoost), dilengkapi analisis eksplorasi data komprehensif, fitur penyimpan artefak model, serta aplikasi web interaktif berbasis Streamlit dengan mode tema Gelap/Terang (*Dark/Light Mode*).

---

## 📋 Deskripsi Proyek

Proyek ini membangun sistem end-to-end prediksi risiko penyakit jantung berbasis dataset `heart_disease_risk_2026.csv` yang memuat **9.000 pasien** dengan **26 fitur** klinis, demografis, dan gaya hidup. Pipeline mencakup seluruh tahapan data science:
- **Eksplorasi Data (EDA)** & Visualisasi Statistik
- **Data Preprocessing**, Encoding, & Feature Scaling
- **Model Training** dengan 5-Fold Stratified Cross-Validation
- **Evaluasi Komprehensif** (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- **Feature Importance Analysis**
- **Inference & Consensus Voting** untuk data pasien baru
- **Serialisasi Artefak Model** ke folder `model/`
- **Interactive Web App Dashboard (Streamlit)** dengan Jarum Gauge Meter & Switcher Tema.

Proyek ini dirancang sebagai portofolio Data Science & Machine Learning.

---

## 📁 Struktur Repositori

```
Heart Disease Risk 2026/
│
├── app.py                         # ← Web Dashboard Streamlit utama
├── requirements.txt               # ← Daftar dependensi Python
├── README.md                      # ← Dokumentasi proyek ini
│
└── Prediction/
    ├── heart_disease.ipynb        # ← Notebook Jupyter (8 Section lengkap)
    ├── heart_disease_risk_2026.csv# ← Dataset pasien (9.000 baris × 26 fitur)
    └── model/                     # ← Folder tempat penyimpanan model & scaler
        ├── model_logistic_regression.pkl
        ├── model_random_forest.pkl
        ├── model_xgboost.pkl
        ├── scaler.pkl
        ├── feature_columns.pkl
        └── model_metadata.json
```

---

## 📊 Dataset

| Atribut | Detail |
|---|---|
| **Dataset** | `heart_disease_risk_2026.csv` |
| **Jumlah Baris** | 9.000 pasien |
| **Jumlah Fitur** | 25 fitur independen + 1 variabel target |
| **Target** | `has_heart_disease` (0 = Tidak Berisiko, 1 = Berisiko Sakit Jantung) |

### Kategori Fitur

| Kategori | Fitur |
|---|---|
| **Demografis** | `age`, `sex` |
| **Tekanan Darah** | `resting_bp_systolic`, `resting_bp_diastolic` |
| **Profil Lipid** | `cholesterol_total`, `hdl`, `ldl`, `triglycerides` |
| **Gula Darah** | `fasting_blood_sugar`, `hba1c` |
| **Antropometri & Kardio** | `bmi`, `resting_heart_rate`, `max_heart_rate_achieved` |
| **Gejala Klinis** | `chest_pain_type`, `exercise_induced_angina`, `st_depression` |
| **Riwayat Medis** | `family_history`, `smoker_status` |
| **Gaya Hidup** | `alcohol_units_per_week`, `exercise_minutes_per_week`, `sleep_hours`, `stress_score`, `wearable_owner`, `daily_steps`, `diet_quality_score` |

---

## 🗂️ Alur Notebook (`heart_disease.ipynb`)

```
1. Import Library
       ↓
2. Load & Eksplorasi Data (EDA)
   ├── Statistik deskriptif
   ├── Distribusi target (bar + pie chart)
   ├── Histogram fitur numerik per kelas
   ├── Bar chart fitur kategorikal vs target
   └── Heatmap korelasi
       ↓
3. Preprocessing
   ├── Cek missing values
   ├── Boolean → int encoding
   ├── Label Encoding (sex)
   ├── One-Hot Encoding (chest_pain_type, smoker_status)
   ├── Train-Test Split 80:20 (stratified)
   └── StandardScaler (untuk Logistic Regression)
       ↓
4. Training 3 Model + 5-Fold Cross-Validation
   ├── Logistic Regression
   ├── Random Forest Classifier
   └── XGBoost Classifier
       ↓
5. Evaluasi Model
   ├── Tabel metrik (Accuracy, Precision, Recall, F1, ROC-AUC, CV AUC)
   ├── Bar chart perbandingan metrik
   ├── Confusion Matrix 3 model
   ├── ROC Curve overlay
   └── Classification Report
       ↓
6. Feature Importance
   ├── Top-15 Random Forest Feature Importance
   └── Top-15 XGBoost Feature Importance
       ↓
7. Inference – Prediksi Data Pasien Baru
   ├── Input data pasien via dictionary
   ├── Preprocessing otomatis (konsisten dengan training)
   ├── Prediksi & probabilitas dari 3 model
   ├── Voting mayoritas (Consensus Voting)
   └── Visualisasi Bar Chart & Gauge Chart
       ↓
8. Simpan Knowledge Model
   ├── Export model .pkl ke folder model/
   ├── Export scaler.pkl & feature_columns.pkl
   ├── Export model_metadata.json
   └── Verifikasi konsistensi prediksi ulang
```

---

## 🤖 Model Machine Learning

| Model | Library | Konfigurasi Utama |
|---|---|---|
| **Logistic Regression** | `scikit-learn` | `max_iter=1000`, `solver='lbfgs'`, Feature Scaled |
| **Random Forest** | `scikit-learn` | `n_estimators=200`, `min_samples_split=5`, `n_jobs=-1` |
| **XGBoost** | `xgboost` | `n_estimators=200`, `learning_rate=0.1`, `max_depth=6` |

Semua model dievaluasi dengan **5-Fold Stratified Cross-Validation** untuk memastikan konsistensi dan mencegah overfitting.

---

## 🌐 Web Application Dashboard (`app.py`)

Aplikasi Web Interaktif berbasis **Streamlit** untuk eksplorasi visualisasi data dan inferensi prediksi risiko secara real-time.

### Fitur Unggulan Web App:
- 🎨 **Dynamic Theme Switcher**: Pilihan **`🌙 Dark Mode`** / **`☀️ Light Mode`** yang mengubah warna antarmuka dan visualisasi secara konsisten.
- 📊 **Dashboard Overview**: KPI Metrics (Total Pasien, Prevalensi, BP, Kolesterol), Pie Chart Prevalensi, Breakdown Usia & Gender, Chest Pain & Merokok, serta Plotly Heatmap Korelasi.
- 🧪 **Analisis Klinis & Gaya Hidup**: 
  - Grafik Scatter SVG (Tekanan Darah, Gula Darah vs HbA1c, Langkah Harian vs Stress Score) yang *bebas error WebGL*.
  - Boxplot Profil Lipid (Kolesterol, HDL, LDL, Triglycerides).
  - Mode Switcher Visualisasi Multivariat: **2D Bubble Chart (SVG)** vs **3D Scatter Plot**.
- 🎯 **Prediksi Real-time Pasien Baru**:
  - Form input interaktif 25 parameter medis & gaya hidup.
  - Prediksi ensemble 3 model ML secara bersamaan.
  - **Consensus Voting** (Keputusan Akhir).
  - **Plotly Semi-Circle Gauge Meter** dengan **Jarum Penunjuk Dinamis (*Needle Arrow Pointer*)**.
- 📋 **Eksplorasi & Export Data**: Global Filters (Rentang Usia, Gender, Status Merokok, Diagnosis) & Tombol Download CSV Terfilter.

---

## 🛠️ Panduan Instalasi & Cara Menjalankan

### 1. Clone Repositori

```bash
git clone https://github.com/<username>/heart-disease-risk-2026.git
cd heart-disease-risk-2026
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Web Application (Streamlit)

```bash
streamlit run app.py
```

### 4. Jalankan Jupyter Notebook (Opsional)

```bash
jupyter notebook Prediction/heart_disease.ipynb
```

---

## 📦 Dependensi Utama (`requirements.txt`)

| Library | Versi | Fungsi |
|---|---|---|
| `streamlit` | ≥ 1.25.0 | Web Application Framework |
| `plotly` | ≥ 5.10.0 | Grafik & Visualisasi Interaktif |
| `scikit-learn` | ≥ 1.1.0 | Preprocessing, Scaling, & Model ML |
| `xgboost` | ≥ 1.7.0 | Gradient Boosting Model |
| `pandas` | ≥ 1.5.0 | Manipulasi & Pengolahan Dataframe |
| `numpy` | ≥ 1.23.0 | Komputasi Numerik |
| `joblib` | ≥ 1.2.0 | Serialisasi / Load Model & Scaler |
| `matplotlib` | ≥ 3.6.0 | Plotting Statistikal Notebook |
| `seaborn` | ≥ 0.12.0 | Styling Grafik Notebook |

---

## 🗺️ Roadmap Proyek

- [x] Exploratory Data Analysis (EDA) komprehensif & visualisasi statistik
- [x] Pipeline Preprocessing & Feature Scaling konsisten
- [x] Training & evaluasi 3 Model Machine Learning
- [x] Inkrementasi Section 8: Serialisasi Artefak Model & Metadata
- [x] Penanganan rendering grafik tanpa WebGL (SVG Fallback)
- [x] Implementasi Gauge Meter dengan Jarum Penunjuk (*Needle Arrow Pointer*)
- [x] Deployment Web Application (Streamlit Dashboard dengan Dark/Light Theme)
- [ ] Hyperparameter Tuning menggunakan Optuna / GridSearchCV
- [ ] Implementasi Model Interpretability (SHAP / LIME Values)

---

## 👤 Author & Portofolio

**Antigravity Pair Programmer**  
📧 *Data Science & Machine Learning Portfolio*

---

## 📄 Lisensi

Proyek ini terlisensi di bawah [MIT License](LICENSE). Silakan gunakan dan kembangkan sesuai kebutuhan.

---

<div align="center">

⭐ **Jika proyek ini membantu atau bermanfaat, jangan lupa berikan Star di GitHub!** ⭐

</div>
