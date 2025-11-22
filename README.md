# vityarthi.project-
# AI: House Price Prediction Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -------------------------
# 1. Create a sample dataset
# -------------------------
data = {
    "area_sqft": [1200, 1500, 800, 2000, 900, 1800, 1700, 1000],
    "bedrooms": [3, 4, 2, 5, 2, 4, 3, 2],
    "age_years": [10, 5, 20, 3, 15, 7, 6, 25],
    "price_lakhs": [60, 85, 40, 120, 45, 95, 90, 35]
}

df = pd.DataFrame(data)

# -------------------------
# 2. Split the data
# -------------------------
X = df[["area_sqft", "bedrooms", "age_years"]]  # Features
y = df["price_lakhs"]                           # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. Train the AI Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# 4. Test the model
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# 5. Evaluate performance
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# -------------------------
# 6. Make your own prediction
# -------------------------
your_house = [[1400, 3, 8]]  # 1400 sqft, 3BHK, 8 years old
predicted_price = model.predict(your_house)[0]
print("Predicted Price (Lakhs):", predicted_price)
