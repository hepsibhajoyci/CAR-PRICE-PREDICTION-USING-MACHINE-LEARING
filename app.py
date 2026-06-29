import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("🚗 Car Price Prediction Dashboard")
st.write("Machine Learning Project - CodeAlpha")

# Load Dataset
df = pd.read_csv("car data.csv")

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Dataset Information")
st.write(df.describe())

# Target Column
target = "Selling_Price"

# Features
X = df.drop(columns=["Selling_Price"])

# Remove Car Name
if "Car_Name" in X.columns:
    X = X.drop(columns=["Car_Name"])

y = df["Selling_Price"]

# Numerical & Categorical Columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

st.subheader("Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", round(mae,2))
c2.metric("MSE", round(mse,2))
c3.metric("RMSE", round(rmse,2))
c4.metric("R² Score", round(r2,2))

# Actual vs Predicted
st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(y_test, pred)

ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)

# Prediction Table
st.subheader("Prediction Results")

result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": pred
})

st.dataframe(result.head(20))

# Sidebar Prediction
st.sidebar.header("Predict Car Price")

year = st.sidebar.number_input("Year",2000,2025,2018)
present_price = st.sidebar.number_input("Present Price",0.0,100.0,5.0)
kms = st.sidebar.number_input("Driven Kms",0,500000,30000)
owner = st.sidebar.selectbox("Owner",[0,1,2,3])
fuel = st.sidebar.selectbox("Fuel Type",["Petrol","Diesel","CNG"])
seller = st.sidebar.selectbox("Seller Type",["Dealer","Individual"])
trans = st.sidebar.selectbox("Transmission",["Manual","Automatic"])

sample = pd.DataFrame({
    "Year":[year],
    "Present_Price":[present_price],
    "Driven_kms":[kms],
    "Fuel_Type":[fuel],
    "Selling_type":[seller],
    "Transmission":[trans],
    "Owner":[owner]
})

if st.sidebar.button("Predict Price"):
    price = model.predict(sample)[0]
    st.sidebar.success(f"Estimated Selling Price: ₹ {price:.2f} Lakhs")