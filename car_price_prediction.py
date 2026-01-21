import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Oasis\TASK 3-Car_Price_Prediction_with_Machine_Learning\car data.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)


# Step 2: Drop unnecessary column
df.drop('Car_Name', axis=1, inplace=True)

print("\nColumns after dropping Car_Name:")
print(df.columns)


# Step 3: Encode categorical variables
encoder = LabelEncoder()

df['Fuel_Type'] = encoder.fit_transform(df['Fuel_Type'])
df['Selling_type'] = encoder.fit_transform(df['Selling_type'])
df['Transmission'] = encoder.fit_transform(df['Transmission'])

print("\nData after encoding categorical variables:")
print(df.head())


# Step 4: Define features (X) and target (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# Step 5: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# Step 6: Train the model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed.")


# Step 7: Make predictions
y_pred = model.predict(X_test)


# Step 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

