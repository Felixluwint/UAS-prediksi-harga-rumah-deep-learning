import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

data = pd.read_csv('../dataset/house_data.csv')
X = data.drop('harga', axis=1)
y = data['harga']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=120, batch_size=8, verbose=0)

# Evaluation
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

# Plot loss
plt.figure()
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('../static/plots/training_loss.png')

# Plot prediction vs actual
plt.figure()
plt.scatter(y_test, pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Prediction vs Actual')
plt.savefig('../static/plots/prediction_vs_actual.png')

model.save('model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("MSE:", mse)
print("R2:", r2)
