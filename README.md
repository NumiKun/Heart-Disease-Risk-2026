# 🫀 Heart Disease Risk Prediction 2026

> **Machine Learning pipeline** untuk memprediksi risiko penyakit jantung menggunakan tiga algoritma klasifikasi, dilengkapi eksplorasi data menyeluruh dan fitur inference data baru.

---

## 📋 Deskripsi Proyek

Proyek ini membangun model prediksi risiko penyakit jantung berbasis dataset sintetis `heart_disease_risk_2026.csv` yang memuat **9.001 pasien** dengan **26 fitur** klinis, demografis, dan gaya hidup. Pipeline mencakup seluruh tahapan data science — mulai dari **EDA**, **preprocessing**, **training 3 model**, **evaluasi komprehensif**, hingga **inference data pasien baru**.

Proyek ini dibuat sebagai bagian dari portofolio data science.

---

## 📁 Struktur Repositori

```
Heart Disease Risk 2026/
│
├── heart_disease.ipynb            # ← Notebook utama (pipeline lengkap)
├── heart_disease_risk_2026.csv    # ← Dataset (9.001 baris × 27 kolom)
└── README.md                      # ← Dokumentasi ini
```

---

## 📊 Dataset

| Atribut | Detail |
|---|---|
| **Sumber** | Dataset sintetis 2026 |
| **Jumlah Baris** | 9.001 pasien |
| **Jumlah Fitur** | 26 fitur + 1 target |
| **Target** | `has_heart_disease` (0 = Tidak, 1 = Ya) |

### Kategori Fitur

| Kategori | Fitur |
|---|---|
| **Demografis** | `age`, `sex` |
| **Tekanan Darah** | `resting_bp_systolic`, `resting_bp_diastolic` |
| **Profil Lipid** | `cholesterol_total`, `hdl`, `ldl`, `triglycerides` |
| **Gula Darah** | `fasting_blood_sugar`, `hba1c` |
| **Antropometri & Kardio** | `bmi`, `resting_heart_rate`, `max_heart_rate_achieved` |
| **Gejala Klinis** | `chest_pain_type`, `exercise_induced_angina`, `st_depression` |
| **Riwayat** | `family_history`, `smoker_status` |
| **Gaya Hidup** | `alcohol_units_per_week`, `exercise_minutes_per_week`, `sleep_hours`, `stress_score`, `wearable_owner`, `daily_steps`, `diet_quality_score` |

---

## 🗂️ Alur Notebook

```
1. Import Library
       ↓
2. Load & EDA
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
       ↓
5. Evaluasi Model
   ├── Tabel metrik (Accuracy, Precision, Recall, F1, ROC-AUC)
   ├── Bar chart perbandingan metrik
   ├── Confusion Matrix (3 model)
   ├── ROC Curve overlay
   └── Classification Report
       ↓
6. Feature Importance
   ├── Top-15 RF Feature Importance
   └── Top-15 XGBoost Feature Importance
       ↓
7. Inference – Prediksi Pasien Baru
   ├── Input data pasien via dictionary
   ├── Preprocessing otomatis (konsisten dengan training)
   ├── Prediksi + probabilitas dari 3 model
   ├── Voting mayoritas
   ├── Bar chart + Gauge chart
   └── Tabel ringkasan hasil
```

---

## 🤖 Model yang Digunakan

| Model | Library | Konfigurasi Utama |
|---|---|---|
| **Logistic Regression** | `sklearn` | `max_iter=1000`, `solver='lbfgs'`, data di-scale |
| **Random Forest** | `sklearn` | `n_estimators=200`, `n_jobs=-1` |
| **XGBoost** | `xgboost` | `n_estimators=200`, `lr=0.1`, `max_depth=6` |

Evaluasi menggunakan **5-Fold Stratified Cross-Validation** untuk estimasi performa yang robust.

---

## 📈 Metrik Evaluasi

Setiap model dievaluasi menggunakan:

- **Accuracy** – proporsi prediksi benar secara keseluruhan
- **Precision** – dari yang diprediksi positif, berapa yang benar-benar positif
- **Recall** – dari yang benar-benar positif, berapa yang berhasil terdeteksi
- **F1-Score** – harmonic mean precision & recall
- **ROC-AUC** – kemampuan diskriminasi model (area under ROC curve)
- **CV AUC** – rata-rata ROC-AUC pada 5-fold cross-validation

---

## 🔬 Fitur Inference Pasien Baru

Section 7 notebook memungkinkan pengguna **memasukkan data pasien baru** dan langsung mendapatkan prediksi dari ketiga model:

```python
data_pasien_baru = {
    'age'                       : 55,
    'sex'                       : 'Male',
    'resting_bp_systolic'       : 140,
    'chest_pain_type'           : 'Asymptomatic',
    'smoker_status'             : 'Former',
    # ... (25 fitur lengkap)
}
```

Output yang dihasilkan:
- ✅ / ⚠️ label prediksi per model
- Probabilitas risiko (%) per model
- **Voting mayoritas** dari 3 model
- **Gauge chart** probabilitas rata-rata
- Tabel ringkasan lengkap

> ⚠️ **Disclaimer:** Prediksi bersifat indikatif dan **bukan pengganti diagnosis medis** dari tenaga kesehatan profesional.

---

## 🌐 Web Application (Streamlit Dashboard)

Aplikasi Web Interaktif berbasis **Streamlit** untuk eksplorasi visualisasi data dan inferensi prediksi risiko secara real-time.

### Fitur Web App:
- 🎨 **Toggle Tema Interaktif**: Pilihan **🌙 Dark Mode** / **☀️ Light Mode**.
- 📊 **Dashboard Overview**: KPI Metrics, Pie/Donut Chart Prevalensi, Breakdown Usia & Gender, Chest Pain & Merokok, serta Plotly Heatmap Korelasi Interaktif.
- 🧪 **Analisis Klinis & Gaya Hidup**: Sebaran Tekanan Darah, Boxplot Profil Lipid, Gula Darah vs HbA1c, Stress vs Langkah Harian, dan Scatter 3D Interaktif.
- 🎯 **Prediksi Real-time Pasien Baru**: Form input parameter klinis lengkap, prediksi ensemble (Logistic Regression, Random Forest, XGBoost), Consensus Voting, dan Gauge Meter Probabilitas Interaktif.
- 📋 **Eksplorasi & Export Data**: Filter data interaktif & opsi download CSV.

---

## 🛠️ Instalasi & Cara Menjalankan

### 1. Clone repositori

```bash
git clone https://github.com/<username>/heart-disease-risk-2026.git
cd heart-disease-risk-2026
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter plotly streamlit joblib
```

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

### 4. Jalankan Notebook (Opsional)

```bash
jupyter notebook Prediction/heart_disease.ipynb
```

---

## 📦 Dependencies

| Library | Versi Minimum | Fungsi |
|---|---|---|
| `streamlit` | ≥ 1.25 | Web App Dashboard |
| `plotly` | ≥ 5.10 | Visualisasi Grafis Interaktif |
| `numpy` | ≥ 1.23 | Komputasi numerik |
| `pandas` | ≥ 1.5 | Manipulasi data |
| `scikit-learn` | ≥ 1.1 | Preprocessing & model ML |
| `xgboost` | ≥ 1.7 | Model XGBoost classifier |
| `joblib` | ≥ 1.2 | Load model & scaler |

---

## 🗺️ Roadmap

- [x] EDA lengkap dengan visualisasi interaktif
- [x] Pipeline preprocessing yang konsisten
- [x] Training & evaluasi 3 model
- [x] Inference data pasien baru dengan voting
- [x] Web Application Dashboard (Streamlit dengan Dark/Light Mode)
- [ ] Hyperparameter tuning (GridSearchCV / Optuna)
- [ ] SHAP values untuk explainability
- [ ] Handling imbalanced class (SMOTE / class_weight)

---

## 👤 Author

**[Nama Kamu]**  
📧 email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/username) | [GitHub](https://github.com/username)

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE). Silakan gunakan dan modifikasi sesuai kebutuhan.

---

<div align="center">

⭐ **Jika proyek ini bermanfaat, jangan lupa beri star!** ⭐

</div>
