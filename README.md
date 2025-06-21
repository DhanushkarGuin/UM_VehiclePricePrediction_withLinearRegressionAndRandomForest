# 🚗 Vehicle Price Prediction using Random Forest

This machine learning project predicts the price of a vehicle based on various features like make, model, mileage, fuel type, etc. The model is built using a **Random Forest Regressor**, and includes preprocessing for both numerical and categorical data through a clean and scalable pipeline.

## 📌 Problem Statement

Predict the **selling price** of used vehicles using historical data. This is a regression task based on structured features like mileage, transmission type, fuel, colors, and engine specs.

## 📊 Dataset

- **Source**: Provided by an Internship Program 
- **Format**: CSV
- **Target variable**: `price`
- **Features include**:
  - Year, Make, Model, Cylinders
  - Fuel Type, Transmission, Mileage
  - Trim, Body Type, Number of Doors
  - Exterior/Interior Color, Drivetrain

## ⚙️ Features & Processing

- **Dropped Columns**: `name`, `description` (to reduce dimensionality and noise)
- **Missing Value Handling**:
  - `mean` imputation for numerical columns like `price`, `mileage`
  - `mode` imputation for categorical columns
- **Preprocessing**:
  - Standardization for numerical data
  - One-hot encoding for categorical data
  - Combined using `ColumnTransformer` and `Pipeline` from scikit-learn

## 🔧 Technologies Used

- Python 3.x
- `pandas`, `numpy` for data handling
- `scikit-learn` for preprocessing and modeling
- `pickle` for saving the trained model

## 🧠 Model

- **Algorithm**: Random Forest Regressor
- **Parameters**: `n_estimators=100`
- **Evaluation Metrics**:
  - Root Mean Squared Error (RMSE)
  - R² Score

### 📈 Results

| Metric | Value |
|--------|-------|
| RMSE   | 7324.05 |
| R²     |  0.78 |

