import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# --- 1. DATA LOADING ---
# Using the path where you are currently working
file_name = 'housing_data.csv'

if not os.path.exists(file_name):
    print(f"❌ Error: {file_name} not found. Ensure you 'cd Syntecxhub' in your terminal.")
else:
    df = pd.read_csv(file_name)
    
    # Standardize column names (remove hidden spaces and handle case)
    df.columns = df.columns.str.strip()
    
    print("✅ Dataset Loaded. Columns found:", df.columns.tolist())

    # --- 2. PREPROCESSING & FEATURE SELECTION ---
    # Automatically filter for numeric columns to avoid "could not convert string to float"
    numeric_df = df.select_dtypes(include=[np.number])
    
    # IDENTIFY TARGET: Update 'Price' if your column is named differently (e.g., 'price' or 'Value')
    target_name = 'price' 
    
    if target_name not in numeric_df.columns:
        # Fallback: check if it's named 'price' or 'value' (case insensitive)
        potential_targets = [col for col in df.columns if col.lower() in ['price', 'value', 'amount']]
        if potential_targets:
            target_name = potential_targets[0]
        else:
            raise KeyError(f"Target column '{target_name}' not found. Please check your CSV header.")

    # Drop rows with missing values
    numeric_df = numeric_df.dropna()

    X = numeric_df.drop(columns=[target_name])
    y = numeric_df[target_name]

    print(f"📊 Using features: {X.columns.tolist()}")
    print(f"🎯 Target variable: {target_name}")

    # --- 3. TRAIN/TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. MODEL TRAINING ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- 5. EVALUATION ---
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Performance ---")
    print(f"RMSE (Error): {rmse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    # --- 6. INTERPRETATION ---
    # Show which features affect the price most
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    print("\nFeature Impact:")
    print(coefficients.sort_values(by='Coefficient', ascending=False))

    # --- 7. SAVE THE MODEL ---
    joblib.dump(model, 'house_price_model.pkl')
    print("\n✅ Project Complete: Model saved as 'house_price_model.pkl'")

    # --- 8. VISUALIZATION (Optional) ---
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.show()