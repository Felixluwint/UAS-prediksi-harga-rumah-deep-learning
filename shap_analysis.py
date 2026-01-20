import numpy as np
import pandas as pd
import shap
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# ===============================
# 1. LOAD MODEL & SCALER
# ===============================
model = tf.keras.models.load_model(
    'model/model.h5',
    compile=False
)
scaler = joblib.load('model/scaler.pkl')

# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv('dataset/house_data.csv')

X = df.drop('harga', axis=1)

# ===============================
# 3. DATA UNTUK SHAP
# ===============================
background = X.sample(min(100, len(X)), random_state=1)
X_explain = X.sample(min(300, len(X)), random_state=2)

background_scaled = scaler.transform(background)
X_explain_scaled = scaler.transform(X_explain)

# ===============================
# 4. SHAP EXPLAINER
# ===============================
explainer = shap.KernelExplainer(
    lambda x: model.predict(x).flatten(),  # 🔥 INI KUNCINYA
    background_scaled
)

shap_values = explainer.shap_values(X_explain_scaled)

# ===============================
# 5. SHAP SUMMARY (FULL FITUR)
# ===============================
shap.summary_plot(
    shap_values,
    X_explain_scaled,
    feature_names=[
        'Luas Tanah',
        'Luas Bangunan',
        'Kamar Tidur',
        'Kamar Mandi',
        'Jarak Pusat Kota',
        'Usia Bangunan'
    ],
    max_display=6,
    show=False
)

# ===============================
# 6. SIMPAN HASIL
# ===============================
plt.tight_layout()
plt.savefig(
    'static/plots/shap_summary.png',
    dpi=150,
    bbox_inches='tight'
)
plt.show()

print("===================================")
print("SHAP FINAL BERHASIL")
print("static/plots/shap_summary.png")
print("===================================")
