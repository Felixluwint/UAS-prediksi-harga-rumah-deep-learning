import numpy as np
import pandas as pd

np.random.seed(42)

N = 800  # jumlah data (boleh 500–1000)

data = {
    "luas_tanah": np.random.randint(60, 300, N),
    "luas_bangunan": np.random.randint(40, 250, N),
    "kamar_tidur": np.random.randint(1, 6, N),
    "kamar_mandi": np.random.randint(1, 4, N),
    "jarak_pusat_kota": np.random.uniform(1, 20, N).round(2),
    "usia_bangunan": np.random.randint(0, 30, N)
}

df = pd.DataFrame(data)

# Rumus harga realistis (dalam juta rupiah)
df["harga"] = (
    df["luas_bangunan"] * 3.2 +
    df["luas_tanah"] * 1.6 +
    df["kamar_tidur"] * 25 +
    df["kamar_mandi"] * 30 -
    df["jarak_pusat_kota"] * 12 -
    df["usia_bangunan"] * 2.5 +
    np.random.normal(0, 35, N)
).round(2)

# Simpan dataset
df.to_csv("dataset/house_data.csv", index=False)

print("Dataset berhasil dibuat!")
print("Jumlah data:", len(df))
print(df.head())
