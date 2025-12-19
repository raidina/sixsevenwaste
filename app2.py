import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


st.title("ระบบทำนายปริมาณขยะ")


df = pd.read_csv('sustainable_waste_management_dataset_2024.csv')

features = ['population', 'temp_c', 'rain_mm', 'is_weekend', 'is_holiday', 'recycling_campaign']
X = df[features]
y = df['waste_kg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R2 Score:** {r2:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel('Actual Waste (kg)')
ax.set_ylabel('Predicted Waste (kg)')
ax.set_title('Actual vs. Predicted Waste (kg)')
ax.grid(True)

st.pyplot(fig)

st.subheader("ตัวอย่างข้อมูล")
st.write(df.head())