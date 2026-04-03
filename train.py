import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("solarpowergeneration.csv")

# Cleaning
df = df.dropna()
df = df.drop(columns=["wind_direction"], errors='ignore')

# 🔥 Handle categorical
if 'sky_cover' in df.columns:
    df = pd.get_dummies(df, columns=['sky_cover'], drop_first=True)

# 🔥 FEATURE ENGINEERING (KEY BOOST)
df['temp_humidity'] = df['temperature'] * df['humidity']
df['wind_pressure'] = df['wind_speed'] * df['average_pressure']
df['temp_squared'] = df['temperature'] ** 2
df['humidity_squared'] = df['humidity'] ** 2
df['wind_squared'] = df['wind_speed'] ** 2

# Features & target
X = df.drop("power_generated", axis=1)
y = df["power_generated"]

# Save columns
model_columns = X.columns

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 TUNED RANDOM FOREST
model = RandomForestRegressor(
    n_estimators=600,          # more trees
    max_depth=25,              # deeper trees
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',       # important for performance
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("-" * 30)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(model_columns, open('columns.pkl', 'wb'))

print(" pickle model is saved!")