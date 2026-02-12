# 🏠 House Price Prediction

The project focuses on building a robust predictive model by analyzing historical housing data. It handles everything from data preprocessing to model evaluation, ensuring that categorical or text-based data doesn't interfere with the mathematical training process.

## 🛠️ Features
- **Smart Preprocessing:** Automatically detects and filters numeric features, ignoring non-computable text descriptions.
- **Evaluation Metrics:** Uses **Root Mean Squared Error (RMSE)** and **$R^2$ Score** for high-precision validation.
- **Visualization:** Includes actual vs. predicted price mapping to visualize model accuracy.
- **Deployment Ready:** Saves the trained model as a `.pkl` file for easy integration into larger applications.

## 📊 Performance Statistics
- **Model:** Linear Regression
- **Framework:** Scikit-Learn
- **Success Metric:** $R^2$ Score provides a clear percentage of the variance explained by the model.

## 📁 Repository Structure
- `housepricepred.py`: Main Python script containing the ML pipeline.
- `house.csv`: The dataset used for training and testing.
- `house_price_model.pkl`: The serialized version of the trained model.
