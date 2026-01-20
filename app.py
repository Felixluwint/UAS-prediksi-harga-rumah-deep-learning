from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

model = tf.keras.models.load_model('model/model.h5')
scaler = joblib.load('model/scaler.pkl')

feature_names = [
    'Luas Tanah','Luas Bangunan','Kamar Tidur',
    'Kamar Mandi','Jarak Pusat Kota','Usia Bangunan'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = low = high = None
    shap_img = None

    if request.method == 'POST':
        data = [
            float(request.form['luas_tanah']),
            float(request.form['luas_bangunan']),
            float(request.form['kamar_tidur']),
            float(request.form['kamar_mandi']),
            float(request.form['jarak_pusat_kota']),
            float(request.form['usia_bangunan'])
        ]

        X = scaler.transform([data])
        pred = model.predict(X)[0][0]

        result = round(pred,2)
        low = round(pred*0.9,2)
        high = round(pred*1.1,2)

        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)

        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        shap_img = 'static/plots/shap_summary.png'
        plt.savefig(shap_img, bbox_inches='tight')

    return render_template(
        'index.html',
        result=result,
        low=low,
        high=high,
        shap_img=shap_img
    )

if __name__ == '__main__':
    app.run(debug=True)
